"""沪铜涨跌概率预测 v3 — 增强特征 + 弱信号过滤。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundamentals import fundamental_prob, fundamental_summary, merge_fundamentals
from lme_ratio import lme_ratio_prob, lme_ratio_summary, merge_lme_ratio
from lookahead import split_known_outcomes
from model_config import load_config
from signal_filter import classify_signal, count_agreement

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ForecastResult:
    as_of_date: str
    close: float
    prob_up_1d: float
    prob_down_1d: float
    prob_up_5d: float
    prob_down_5d: float
    expected_ret_1d_pct: float
    expected_ret_5d_pct: float
    signal_1d: str = "观望"
    signal_5d: str = "观望"
    signal_1d_reason: str = ""
    signal_5d_reason: str = ""
    signals: dict = field(default_factory=dict)
    disclaimer: str = ""


TECH_COLS_1D = [
    "ret1", "ret5", "ma5_bias", "ma20_bias", "ma_cross", "rsi14",
    "vol20", "oi_chg5", "atr_pct", "macd_hist", "price_pos20",
]
TECH_COLS_5D = [
    "ret5", "ret20", "ma20_bias", "ma60_bias", "ma_cross", "rsi14",
    "vol20", "oi_chg5", "oi_vol_ratio_chg5", "atr_pct", "macd_hist", "price_pos20",
]
FUND_COLS = ["wsr_pct5", "wsr_chg5_norm", "net_position_chg5_norm", "ratio_z60", "ratio_chg5"]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].pct_change()
    out["ret5"] = out["close"].pct_change(5)
    out["ret20"] = out["close"].pct_change(20)
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma60"] = out["close"].rolling(60).mean()
    out["ma5_bias"] = out["close"] / out["ma5"] - 1
    out["ma20_bias"] = out["close"] / out["ma20"] - 1
    out["ma60_bias"] = out["close"] / out["ma60"] - 1
    out["ma_cross"] = (out["ma5"] > out["ma20"]).astype(float)
    out["rsi14"] = _rsi(out["close"], 14)

    tr = pd.concat(
        [out["high"] - out["low"], (out["high"] - out["close"].shift()).abs(), (out["low"] - out["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr14"] / out["close"]
    out["vol20"] = out["ret1"].rolling(20).std()
    out["oi_chg5"] = out["oi"].pct_change(5)
    out["vol_z"] = out["vol"].pct_change(5)

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()) / out["close"]

    rh = out["high"].rolling(20).max()
    rl = out["low"].rolling(20).min()
    out["price_pos20"] = (out["close"] - rl) / (rh - rl + 1e-9)

    out["oi_vol_ratio"] = out["oi"] / out["vol"].replace(0, np.nan)
    out["oi_vol_ratio_chg5"] = out["oi_vol_ratio"].pct_change(5)

    out["next_ret1"] = out["close"].shift(-1) / out["close"] - 1
    out["next_ret5"] = out["close"].shift(-5) / out["close"] - 1
    out["next_up1"] = (out["next_ret1"] > 0).astype(int)
    out["next_up5"] = (out["next_ret5"] > 0).astype(int)
    return out


def _enrich_fundamental_features(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    if "wsr_vol" in out.columns:
        wsr_std = out["wsr_vol"].rolling(60, min_periods=20).std().replace(0, np.nan)
        out["wsr_chg5_norm"] = out.get("wsr_chg5", pd.Series(index=out.index)) / wsr_std
    if "net_position_chg5" in out.columns and "long_total" in out.columns:
        denom = out["long_total"].rolling(20, min_periods=5).mean().replace(0, np.nan)
        out["net_position_chg5_norm"] = out["net_position_chg5"] / denom
    return out


def _active_feature_cols(feat: pd.DataFrame, horizon: int) -> list[str]:
    base = TECH_COLS_1D if horizon == 1 else TECH_COLS_5D
    cols = [c for c in base if c in feat.columns]
    for c in FUND_COLS:
        if c in feat.columns and feat[c].notna().sum() > 50:
            cols.append(c)
    return cols


def _similar_state_prob(feat: pd.DataFrame, horizon: int = 1, end_idx: int | None = None) -> float:
    target = "next_up1" if horizon == 1 else "next_up5"
    cols = _active_feature_cols(feat, horizon)
    usable = feat.dropna(subset=cols + [target]).copy()
    latest, hist = split_known_outcomes(usable, horizon, end_idx)
    if len(hist) < 120:
        return 0.5
    dist = np.zeros(len(hist))
    for col in cols:
        std = hist[col].std()
        if std and not pd.isna(std):
            dist += ((hist[col] - latest[col]) / std) ** 2
    dist = np.sqrt(dist / max(len(cols), 1))
    k = max(30, int(len(hist) * 0.12))
    return float(hist.iloc[np.argsort(dist.values)[:k]][target].mean())


def _logistic_prob(feat: pd.DataFrame, horizon: int = 1, end_idx: int | None = None) -> float | None:
    if not HAS_SKLEARN:
        return None
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


def _momentum_vote(row: pd.Series, horizon: int = 1) -> float:
    votes: list[float] = []
    if pd.notna(row.get("ma_cross")):
        votes.append(0.57 if row["ma_cross"] == 1 else 0.43)
    if pd.notna(row.get("rsi14")):
        if horizon == 1:
            if row["rsi14"] < 30:
                votes.append(0.63)
            elif row["rsi14"] > 70:
                votes.append(0.37)
            else:
                votes.append(0.50)
        else:
            if row["rsi14"] < 40:
                votes.append(0.58)
            elif row["rsi14"] > 70:
                votes.append(0.42)
            else:
                votes.append(0.50)
    if pd.notna(row.get("macd_hist")):
        votes.append(0.55 if row["macd_hist"] > 0 else 0.45)
    if pd.notna(row.get("ret5")):
        votes.append(0.54 if row["ret5"] > 0 else 0.46)
    if horizon == 5 and pd.notna(row.get("ret20")):
        votes.append(0.53 if row["ret20"] > 0 else 0.47)
    return float(np.mean(votes)) if votes else 0.5


def _monte_carlo_week(feat: pd.DataFrame, end_idx: int | None = None, n_sim: int = 8000) -> tuple[float, float]:
    sub = feat.iloc[: end_idx + 1] if end_idx is not None else feat
    rets = sub["ret1"].dropna().tail(60).values
    if len(rets) < 30:
        return 0.5, 0.0
    mu, sigma = rets.mean(), rets.std(ddof=1)
    if sigma == 0:
        return (0.55, 0.0) if mu > 0 else (0.45, 0.0)
    sims = np.random.default_rng(42).normal(mu, sigma, size=(n_sim, 5)).sum(axis=1)
    return float((sims > 0).mean()), float(np.median(sims) * 100)


def _apply_regime_adjust(prob: float, row: pd.Series, horizon: int) -> float:
    """极端位置均值回归微调，避免超买区盲目追多。"""
    adj = prob
    if horizon == 1 and pd.notna(row.get("rsi14")) and row["rsi14"] > 75:
        adj -= 0.03
    if horizon == 1 and pd.notna(row.get("price_pos20")) and row["price_pos20"] > 0.92:
        adj -= 0.02
    if horizon == 5 and pd.notna(row.get("rsi14")) and row["rsi14"] < 25:
        adj += 0.02
    return float(np.clip(adj, 0.05, 0.95))


def compute_probs(feat: pd.DataFrame, end_idx: int | None = None, use_fundamentals: bool = True) -> dict:
    if end_idx is None:
        end_idx = len(feat) - 1
    sub = feat.iloc[: end_idx + 1]
    row = sub.iloc[-1]

    sim1 = _similar_state_prob(sub, 1, end_idx)
    sim5 = _similar_state_prob(sub, 5, end_idx)
    log1 = _logistic_prob(sub, 1, end_idx)
    log5 = _logistic_prob(sub, 5, end_idx)
    mom1 = _momentum_vote(row, 1)
    mom5 = _momentum_vote(row, 5)
    mc5, mc_exp5 = _monte_carlo_week(sub, end_idx)
    fund = fundamental_prob(row) if use_fundamentals else 0.5
    lme = lme_ratio_prob(row) if pd.notna(row.get("ratio_z60")) else 0.5

    parts1: list[tuple[float, float]] = [(sim1, 0.20), (mom1, 0.20), (fund, 0.15)]
    if log1 is not None:
        parts1 += [(log1, 0.45)]
    else:
        parts1[0] = (parts1[0][0], 0.35)
    p1 = sum(p * w for p, w in parts1) / sum(w for _, w in parts1)

    # v4: 加入沪伦比；提高 MC + 逻辑回归权重
    parts5: list[tuple[float, float]] = [(mc5, 0.30), (fund, 0.15), (lme, 0.15), (sim5, 0.05)]
    if log5 is not None:
        parts5 += [(log5, 0.30)]
    else:
        parts5[0] = (parts5[0][0], parts5[0][1] + 0.15)
    parts5.append((mom5, 0.05))
    p5 = sum(p * w for p, w in parts5) / sum(w for _, w in parts5)

    p1 = _apply_regime_adjust(p1, row, 1)
    p5 = _apply_regime_adjust(p5, row, 5)

    return {
        "prob_up_1d": p1,
        "prob_up_5d": p5,
        "sim1": sim1, "sim5": sim5,
        "log1": log1, "log5": log5,
        "mom1": mom1, "mom5": mom5,
        "mc5": mc5, "fund": fund, "lme": lme, "mc_exp5": mc_exp5,
    }


def prepare_features(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    lme_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feat = build_feature_frame(df)
    wsr_part = wsr_daily if wsr_daily is not None else pd.DataFrame()
    hold_part = hold_daily if hold_daily is not None else pd.DataFrame()
    if not wsr_part.empty or not hold_part.empty:
        feat = merge_fundamentals(feat, wsr_part, hold_part)
        feat = _enrich_fundamental_features(feat)
    lme_part = lme_daily if lme_daily is not None else pd.DataFrame()
    if not lme_part.empty:
        feat = merge_lme_ratio(feat, lme_part)
    return feat


def forecast_cu(
    df: pd.DataFrame,
    wsr_daily: pd.DataFrame | None = None,
    hold_daily: pd.DataFrame | None = None,
    lme_daily: pd.DataFrame | None = None,
    end_idx: int | None = None,
    cfg: dict | None = None,
) -> ForecastResult:
    feat = prepare_features(df, wsr_daily, hold_daily, lme_daily)
    if end_idx is None:
        end_idx = len(feat) - 1
    return _forecast_from_feat(feat, end_idx, cfg)


def _forecast_from_feat(feat: pd.DataFrame, end_idx: int, cfg: dict | None) -> ForecastResult:
    probs = compute_probs(feat, end_idx, use_fundamentals=True)
    row = feat.iloc[end_idx]
    cfg = cfg or load_config()
    long_only = cfg.get("long_only", True)
    f1 = cfg.get("filter", {}).get("1d", {})
    f5 = cfg.get("filter", {}).get("5d", {})

    sig1, reason1 = classify_signal(
        probs["prob_up_1d"], probs, 1,
        f1.get("th_up", 0.62), f1.get("th_down", 0.38), f1.get("min_agree", 3),
        long_only=long_only,
    )
    sig5, reason5 = classify_signal(
        probs["prob_up_5d"], probs, 5,
        f5.get("th_up", 0.54), f5.get("th_down", 0.45), f5.get("min_agree", 3),
        long_only=long_only,
    )
    b1, _ = count_agreement(probs, 1)
    b5, _ = count_agreement(probs, 5)
    signals = {
        "逻辑回归_1日": round(probs["log1"] * 100, 1) if probs["log1"] else None,
        "逻辑回归_5日": round(probs["log5"] * 100, 1) if probs["log5"] else None,
        "相似形态_5日": round(probs["sim5"] * 100, 1),
        "蒙特卡洛_5日": round(probs["mc5"] * 100, 1),
        "基本面_上涨率": round(probs["fund"] * 100, 1),
        "沪伦比_上涨率": round(probs["lme"] * 100, 1),
        "子模型看多_1日": f"{b1}/4",
        "子模型看多_5日": f"{b5}/5",
        "模式": "仅做多" if long_only else "多空",
        "RSI14": round(float(row["rsi14"]), 1) if pd.notna(row.get("rsi14")) else None,
        "MA5/MA20": "多头排列" if row.get("ma_cross") == 1 else "空头排列",
        "近5日涨跌": round(float(row["ret5"]) * 100, 2) if pd.notna(row.get("ret5")) else None,
        "20日区间位置": round(float(row["price_pos20"]) * 100, 1) if pd.notna(row.get("price_pos20")) else None,
    }
    signals.update(fundamental_summary(row))
    signals.update(lme_ratio_summary(row))
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
        disclaimer="统计概率仅供参考，不构成投资建议；v4仅做多模式，「观望」= 不操作。",
    )


def format_report(result: ForecastResult) -> str:
    lines = [
        "=" * 52,
        "  沪铜(CU) 涨跌概率预测 v4",
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
