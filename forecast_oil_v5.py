"""原油(SC) 涨跌概率预测 v5 — 独立模块，不修改沪铜代码。"""
from __future__ import annotations

import pandas as pd

import forecast as v4
from forecast import ForecastResult
from fundamentals import aggregate_holding, aggregate_wsr, fundamental_prob, fundamental_summary, merge_fundamentals
from lookahead import split_known_outcomes
from macro_features import fetch_cn_pmi_series
from oil_external import fetch_sc_wti_ratio_series, fetch_us_api_crude_series, merge_oil_external, oil_macro_prob, oil_macro_summary, oil_ratio_prob
from model_config_oil import load_config
from quality_filter import apply_quality_to_signal
from signal_filter import classify_signal, count_agreement

MACRO_COLS = ["wti_ret1", "wti_ret5", "us_api_chg5", "ratio_z60", "ratio_chg5", "pmi_mfg", "pmi_chg1"]
V5_EXTRA_5D_KEYS = ["macro"]


def _active_feature_cols(feat: pd.DataFrame, horizon: int) -> list[str]:
    cols = v4._active_feature_cols(feat, horizon)
    for c in MACRO_COLS:
        if c in feat.columns and feat[c].notna().sum() > 50 and c not in cols:
            cols.append(c)
    return cols


def _similar_state_prob(feat: pd.DataFrame, horizon: int = 1, end_idx: int | None = None) -> float:
    import numpy as np

    target = "next_up1" if horizon == 1 else "next_up5"
    cols = _active_feature_cols(feat, horizon)
    usable = feat.dropna(subset=cols + [target]).copy()
    latest, hist = split_known_outcomes(usable, horizon, end_idx)
    if len(hist) < 100:
        return 0.5
    dist = np.zeros(len(hist))
    for col in cols:
        std = hist[col].std()
        if std and not pd.isna(std):
            dist += ((hist[col] - latest[col]) / std) ** 2
    dist = np.sqrt(dist / max(len(cols), 1))
    k = max(25, int(len(hist) * 0.12))
    return float(hist.iloc[np.argsort(dist.values)[:k]][target].mean())


def _logistic_prob(feat: pd.DataFrame, horizon: int = 1, end_idx: int | None = None) -> float | None:
    if not v4.HAS_SKLEARN:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    target = "next_up1" if horizon == 1 else "next_up5"
    cols = _active_feature_cols(feat, horizon)
    usable = feat.dropna(subset=cols + [target]).copy()
    if end_idx is not None:
        usable = usable.iloc[: end_idx + 1]
    if len(usable) <= horizon or len(usable) < 200:
        return None
    train = usable.iloc[: len(usable) - horizon]
    latest_x = usable.iloc[[-1]][cols]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train[cols].values)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    model.fit(x_scaled, train[target].values)
    return float(model.predict_proba(scaler.transform(latest_x.values))[0, 1])


def compute_probs(feat: pd.DataFrame, end_idx: int | None = None, use_fundamentals: bool = True) -> dict:
    if end_idx is None:
        end_idx = len(feat) - 1
    sub = feat.iloc[: end_idx + 1]
    row = sub.iloc[-1]
    sim1 = _similar_state_prob(sub, 1, end_idx)
    sim5 = _similar_state_prob(sub, 5, end_idx)
    log1 = _logistic_prob(sub, 1, end_idx)
    log5 = _logistic_prob(sub, 5, end_idx)
    mom1 = v4._momentum_vote(row, 1)
    mom5 = v4._momentum_vote(row, 5)
    mc5, mc_exp5 = v4._monte_carlo_week(sub, end_idx)
    fund = fundamental_prob(row) if use_fundamentals else 0.5
    ratio = oil_ratio_prob(row) if pd.notna(row.get("ratio_z60")) else 0.5
    macro = oil_macro_prob(row) if pd.notna(row.get("wti_ret1")) or pd.notna(row.get("pmi_mfg")) else 0.5

    parts1: list[tuple[float, float]] = [(sim1, 0.18), (mom1, 0.18), (fund, 0.14), (macro, 0.10)]
    if log1 is not None:
        parts1 += [(log1, 0.40)]
    else:
        parts1[0] = (parts1[0][0], 0.30)
    p1 = sum(p * w for p, w in parts1) / sum(w for _, w in parts1)

    parts5: list[tuple[float, float]] = [(mc5, 0.24), (fund, 0.12), (ratio, 0.12), (macro, 0.12), (sim5, 0.05)]
    if log5 is not None:
        parts5 += [(log5, 0.28)]
    else:
        parts5[0] = (parts5[0][0], parts5[0][1] + 0.13)
    parts5.append((mom5, 0.07))
    p5 = sum(p * w for p, w in parts5) / sum(w for _, w in parts5)
    p1 = v4._apply_regime_adjust(p1, row, 1)
    p5 = v4._apply_regime_adjust(p5, row, 5)
    return {
        "prob_up_1d": p1, "prob_up_5d": p5,
        "sim1": sim1, "sim5": sim5, "log1": log1, "log5": log5,
        "mom1": mom1, "mom5": mom5, "mc5": mc5, "fund": fund, "ratio": ratio, "macro": macro, "mc_exp5": mc_exp5,
    }


def prepare_features(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    ratio_daily: pd.DataFrame | None = None,
    api_daily: pd.DataFrame | None = None,
    pmi_monthly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feat = v4.build_feature_frame(df)
    wsr = wsr_daily if wsr_daily is not None else pd.DataFrame()
    hold = hold_daily if hold_daily is not None else pd.DataFrame()
    if not wsr.empty or not hold.empty:
        feat = merge_fundamentals(feat, wsr, hold)
        feat = v4._enrich_fundamental_features(feat)
    ratio = ratio_daily if ratio_daily is not None else pd.DataFrame()
    api = api_daily if api_daily is not None else pd.DataFrame()
    return merge_oil_external(feat, ratio, api, pmi_monthly)


def forecast_sc(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    ratio_daily: pd.DataFrame | None = None,
    api_daily: pd.DataFrame | None = None,
    pmi_monthly: pd.DataFrame | None = None,
    end_idx: int | None = None,
    cfg: dict | None = None,
) -> ForecastResult:
    feat = prepare_features(df, wsr_daily, hold_daily, ratio_daily, api_daily, pmi_monthly)
    if end_idx is None:
        end_idx = len(feat) - 1
    return _forecast_from_feat(feat, end_idx, cfg)


def _forecast_from_feat(feat: pd.DataFrame, end_idx: int, cfg: dict | None) -> ForecastResult:
    probs = compute_probs(feat, end_idx, use_fundamentals=True)
    row = feat.iloc[end_idx]
    cfg = cfg or load_config()
    f1, f5 = cfg.get("filter", {}).get("1d", {}), cfg.get("filter", {}).get("5d", {})
    extra = V5_EXTRA_5D_KEYS
    sig1, reason1 = classify_signal(probs["prob_up_1d"], probs, 1, f1.get("th_up", 0.62), f1.get("th_down", 0.38), f1.get("min_agree", 3), long_only=True)
    sig5, reason5 = classify_signal(probs["prob_up_5d"], probs, 5, f5.get("th_up", 0.57), f5.get("th_down", 0.44), f5.get("min_agree", 4), long_only=True, extra_5d_keys=extra)
    sig5, reason5 = apply_quality_to_signal(sig5, reason5, row, cfg.get("quality_filter"))
    b5 = count_agreement(probs, 5, extra_5d_keys=extra)[0]
    max5 = 5 + len(extra)
    signals = {
        "蒙特卡洛_5日": round(probs["mc5"] * 100, 1),
        "基本面_上涨率": round(probs["fund"] * 100, 1),
        "SC/WTI比_上涨率": round(probs["ratio"] * 100, 1),
        "宏观_上涨率": round(probs["macro"] * 100, 1),
        "子模型看多_5日": f"{b5}/{max5}",
        "RSI14": round(float(row["rsi14"]), 1) if pd.notna(row.get("rsi14")) else None,
        "MA5/MA20": "多头排列" if row.get("ma_cross") == 1 else "空头排列",
        "20日区间位置": round(float(row["price_pos20"]) * 100, 1) if pd.notna(row.get("price_pos20")) else None,
    }
    signals.update(fundamental_summary(row))
    signals.update(oil_macro_summary(row))
    return ForecastResult(
        as_of_date=str(row["trade_date"]),
        close=float(row["close"]),
        prob_up_1d=round(probs["prob_up_1d"], 4),
        prob_down_1d=round(1 - probs["prob_up_1d"], 4),
        prob_up_5d=round(probs["prob_up_5d"], 4),
        prob_down_5d=round(1 - probs["prob_up_5d"], 4),
        expected_ret_1d_pct=0.0,
        expected_ret_5d_pct=round(probs["mc_exp5"], 3),
        signal_1d=sig1, signal_5d=sig5,
        signal_1d_reason=reason1, signal_5d_reason=reason5,
        signals=signals,
        disclaimer="原油统计概率仅供参考；v5仅做多模式。",
    )


def load_oil_data(start: str, use_cache: bool = True):
    from fetcher_oil import fetch_sc_daily, fetch_sc_holding_series, fetch_sc_wsr_series

    df = fetch_sc_daily(start_date=start, use_cache=use_cache)
    wsr = fetch_sc_wsr_series(start, use_cache=use_cache)
    hold = fetch_sc_holding_series(start, use_cache=use_cache)
    ratio = fetch_sc_wti_ratio_series(df, start, use_cache=use_cache)
    api = fetch_us_api_crude_series(start, use_cache=use_cache)
    pmi = fetch_cn_pmi_series(start, use_cache=use_cache)
    return df, aggregate_wsr(wsr), aggregate_holding(hold), ratio, api, pmi


def format_report(result: ForecastResult) -> str:
    sig = result.signals
    lines = [
        "=" * 52,
        f"  原油(SC) 涨跌概率预测 v5  |  截至 {result.as_of_date}",
        "=" * 52,
        f"  收盘价: {result.close:,.1f} 元/桶",
        "",
        f"  下一交易日  涨 {result.prob_up_1d*100:.1f}%  跌 {result.prob_down_1d*100:.1f}%  [{result.signal_1d}]",
        f"  未来一周    涨 {result.prob_up_5d*100:.1f}%  跌 {result.prob_down_5d*100:.1f}%  [{result.signal_5d}]",
        "",
        f"  子模型: 蒙特卡洛{sig.get('蒙特卡洛_5日')}% 基本面{sig.get('基本面_上涨率')}% "
        f"SC/WTI{sig.get('SC/WTI比_上涨率')}% 宏观{sig.get('宏观_上涨率')}% ({sig.get('子模型看多_5日')})",
        "",
        f"  {result.disclaimer}",
        "=" * 52,
    ]
    return "\n".join(lines)
