"""沪铜涨跌概率预测 v5 — 在 v4 基础上增加 LME涨跌/库存 + PMI，不修改 v4 代码。"""
from __future__ import annotations

import pandas as pd

import forecast as v4
from forecast import ForecastResult
from lookahead import split_known_outcomes
from macro_features import macro_prob, macro_summary, merge_macro_features
from model_config import load_config
from quality_filter import apply_quality_to_signal
from signal_filter import classify_signal, count_agreement

MACRO_COLS = ["lme_ret1", "lme_ret5", "lme_inv_chg5", "lme_inv_z60", "pmi_mfg", "pmi_chg1"]
V5_EXTRA_5D_KEYS = ["macro"]


def _active_feature_cols(feat: pd.DataFrame, horizon: int) -> list[str]:
    cols = v4._active_feature_cols(feat, horizon)
    for c in MACRO_COLS:
        if c in feat.columns and feat[c].notna().sum() > 50 and c not in cols:
            cols.append(c)
    return cols


def _similar_state_prob(feat: pd.DataFrame, horizon: int = 1, end_idx: int | None = None) -> float:
    target = "next_up1" if horizon == 1 else "next_up5"
    cols = _active_feature_cols(feat, horizon)
    usable = feat.dropna(subset=cols + [target]).copy()
    latest, hist = split_known_outcomes(usable, horizon, end_idx)
    if len(hist) < 120:
        return 0.5
    import numpy as np

    dist = np.zeros(len(hist))
    for col in cols:
        std = hist[col].std()
        if std and not pd.isna(std):
            dist += ((hist[col] - latest[col]) / std) ** 2
    dist = np.sqrt(dist / max(len(cols), 1))
    k = max(30, int(len(hist) * 0.12))
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
    if len(usable) <= horizon:
        return None
    train = usable.iloc[: len(usable) - horizon]
    latest_x = usable.iloc[[-1]][cols]
    if len(train) < 250:
        return None
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
    fund = v4.fundamental_prob(row) if use_fundamentals else 0.5
    lme = v4.lme_ratio_prob(row) if pd.notna(row.get("ratio_z60")) else 0.5
    macro = macro_prob(row) if pd.notna(row.get("lme_ret1")) or pd.notna(row.get("pmi_mfg")) else 0.5

    parts1: list[tuple[float, float]] = [(sim1, 0.18), (mom1, 0.18), (fund, 0.14), (macro, 0.10)]
    if log1 is not None:
        parts1 += [(log1, 0.40)]
    else:
        parts1[0] = (parts1[0][0], 0.30)
    p1 = sum(p * w for p, w in parts1) / sum(w for _, w in parts1)

    parts5: list[tuple[float, float]] = [
        (mc5, 0.24),
        (fund, 0.12),
        (lme, 0.12),
        (macro, 0.12),
        (sim5, 0.05),
    ]
    if log5 is not None:
        parts5 += [(log5, 0.28)]
    else:
        parts5[0] = (parts5[0][0], parts5[0][1] + 0.13)
    parts5.append((mom5, 0.07))
    p5 = sum(p * w for p, w in parts5) / sum(w for _, w in parts5)

    p1 = v4._apply_regime_adjust(p1, row, 1)
    p5 = v4._apply_regime_adjust(p5, row, 5)

    return {
        "prob_up_1d": p1,
        "prob_up_5d": p5,
        "sim1": sim1,
        "sim5": sim5,
        "log1": log1,
        "log5": log5,
        "mom1": mom1,
        "mom5": mom5,
        "mc5": mc5,
        "fund": fund,
        "lme": lme,
        "macro": macro,
        "mc_exp5": mc_exp5,
    }


def prepare_features(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    lme_daily: pd.DataFrame | None = None,
    lme_inventory: pd.DataFrame | None = None,
    pmi_monthly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feat = v4.prepare_features(df, wsr_daily, hold_daily, lme_daily)
    return merge_macro_features(feat, lme_inventory, pmi_monthly)


def forecast_cu(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    lme_daily: pd.DataFrame | None = None,
    lme_inventory: pd.DataFrame | None = None,
    pmi_monthly: pd.DataFrame | None = None,
    end_idx: int | None = None,
    cfg: dict | None = None,
) -> ForecastResult:
    feat = prepare_features(df, wsr_daily, hold_daily, lme_daily, lme_inventory, pmi_monthly)
    if end_idx is None:
        end_idx = len(feat) - 1
    return _forecast_from_feat(feat, end_idx, cfg)


def _forecast_from_feat(feat: pd.DataFrame, end_idx: int, cfg: dict | None) -> ForecastResult:
    probs = compute_probs(feat, end_idx, use_fundamentals=True)
    row = feat.iloc[end_idx]
    cfg = cfg or load_config(version=5)
    long_only = cfg.get("long_only", True)
    f1 = cfg.get("filter", {}).get("1d", {})
    f5 = cfg.get("filter", {}).get("5d", {})
    extra = V5_EXTRA_5D_KEYS

    sig1, reason1 = classify_signal(
        probs["prob_up_1d"],
        probs,
        1,
        f1.get("th_up", 0.62),
        f1.get("th_down", 0.38),
        f1.get("min_agree", 3),
        long_only=long_only,
    )
    sig5, reason5 = classify_signal(
        probs["prob_up_5d"],
        probs,
        5,
        f5.get("th_up", 0.57),
        f5.get("th_down", 0.45),
        f5.get("min_agree", 4),
        long_only=long_only,
        extra_5d_keys=extra,
    )
    sig5, reason5 = apply_quality_to_signal(sig5, reason5, row, cfg.get("quality_filter"))
    b1, _ = count_agreement(probs, 1)
    b5, _ = count_agreement(probs, 5, extra_5d_keys=extra)
    max5 = 5 + len(extra)
    signals = {
        "逻辑回归_1日": round(probs["log1"] * 100, 1) if probs["log1"] else None,
        "逻辑回归_5日": round(probs["log5"] * 100, 1) if probs["log5"] else None,
        "相似形态_5日": round(probs["sim5"] * 100, 1),
        "蒙特卡洛_5日": round(probs["mc5"] * 100, 1),
        "基本面_上涨率": round(probs["fund"] * 100, 1),
        "沪伦比_上涨率": round(probs["lme"] * 100, 1),
        "宏观_上涨率": round(probs["macro"] * 100, 1),
        "子模型看多_1日": f"{b1}/4",
        "子模型看多_5日": f"{b5}/{max5}",
        "模式": "仅做多" if long_only else "多空",
        "RSI14": round(float(row["rsi14"]), 1) if pd.notna(row.get("rsi14")) else None,
        "MA5/MA20": "多头排列" if row.get("ma_cross") == 1 else "空头排列",
        "近5日涨跌": round(float(row["ret5"]) * 100, 2) if pd.notna(row.get("ret5")) else None,
        "20日区间位置": round(float(row["price_pos20"]) * 100, 1) if pd.notna(row.get("price_pos20")) else None,
    }
    signals.update(v4.fundamental_summary(row))
    signals.update(v4.lme_ratio_summary(row))
    signals.update(macro_summary(row))
    exp_1d = float(feat["next_ret1"].dropna().mean() * 100) if feat["next_ret1"].notna().any() else 0.0
    exp_5d = float(feat["next_ret5"].dropna().mean() * 100) if feat["next_ret5"].notna().any() else 0.0
    return ForecastResult(
        as_of_date=str(row["trade_date"]),
        close=float(row["close"]),
        prob_up_1d=round(probs["prob_up_1d"], 4),
        prob_down_1d=round(1 - probs["prob_up_1d"], 4),
        prob_up_5d=round(probs["prob_up_5d"], 4),
        prob_down_5d=round(1 - probs["prob_up_5d"], 4),
        expected_ret_1d_pct=round(exp_1d, 3),
        expected_ret_5d_pct=round((probs["mc_exp5"] + exp_5d) / 2, 3),
        signal_1d=sig1,
        signal_5d=sig5,
        signal_1d_reason=reason1,
        signal_5d_reason=reason5,
        signals=signals,
        disclaimer="统计概率仅供参考，不构成投资建议；v5仅做多模式，「观望」= 不操作。",
    )


def format_report(result: ForecastResult) -> str:
    lines = [
        "=" * 52,
        "  沪铜(CU) 涨跌概率预测 v5",
        "=" * 52,
        f"  数据截止     : {result.as_of_date}",
        f"  最新收盘价   : {result.close:,.0f} 元/吨",
        "",
        "  【下一交易日】",
        f"  上涨概率     : {result.prob_up_1d * 100:.1f}%",
        f"  下跌概率     : {result.prob_down_1d * 100:.1f}%",
        f"  ★ 操作建议   : {result.signal_1d}  ({result.signal_1d_reason})",
        "",
        "  【未来一周 ≈ 5个交易日】",
        f"  上涨概率     : {result.prob_up_5d * 100:.1f}%",
        f"  下跌概率     : {result.prob_down_5d * 100:.1f}%",
        f"  ★ 操作建议   : {result.signal_5d}  ({result.signal_5d_reason})",
        "",
        "  【分项信号】",
    ]
    for k, v in result.signals.items():
        if v is not None:
            lines.append(f"  {k:16s}: {v}")
    lines.extend(["", f"  {result.disclaimer}", "=" * 52])
    return "\n".join(lines)
