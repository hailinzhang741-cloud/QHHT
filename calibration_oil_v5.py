"""原油(SC) 概率校准回测 v5。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calibration import calibration_metrics
from forecast_oil_v5 import compute_probs, prepare_features
from quality_filter import passes_quality
from signal_filter import count_agreement

V5_EXTRA_5D_KEYS = ["macro"]


def run_calibration_backtest(
    feat: pd.DataFrame,
    min_train: int = 252,
    step: int = 3,
    horizon: int = 1,
) -> pd.DataFrame:
    target_col = "next_up1" if horizon == 1 else "next_up5"
    rows: list[dict] = []

    for idx in range(min_train, len(feat) - horizon, step):
        row = feat.iloc[idx]
        if pd.isna(row.get(target_col)):
            continue

        probs = compute_probs(feat, idx, use_fundamentals=True)
        key = "prob_up_1d" if horizon == 1 else "prob_up_5d"
        rows.append(
            {
                "trade_date": row["trade_date"],
                "close": row["close"],
                "pred_prob_up": probs[key],
                "actual_up": int(row[target_col]),
                "pred_label": 1 if probs[key] >= 0.5 else 0,
                "correct": int((probs[key] >= 0.5) == int(row[target_col])),
                "horizon": horizon,
                "probs_dict": probs,
                "rsi14": row.get("rsi14"),
                "price_pos20": row.get("price_pos20"),
                "ret5": row.get("ret5"),
            }
        )

    return pd.DataFrame(rows)


def eval_filter_v5(
    cal_df: pd.DataFrame,
    th_up: float,
    min_agree: int,
    quality_cfg: dict | None,
    long_only: bool = True,
) -> dict:
    long_hits = long_n = 0
    for _, r in cal_df.iterrows():
        probs = r.get("probs_dict", {})
        bull, _ = count_agreement(probs, 5, extra_5d_keys=V5_EXTRA_5D_KEYS)
        p, actual = r["pred_prob_up"], r["actual_up"]
        if p < th_up or bull < min_agree:
            continue
        row = pd.Series({"rsi14": r.get("rsi14"), "price_pos20": r.get("price_pos20"), "ret5": r.get("ret5")})
        ok, _ = passes_quality(row, quality_cfg)
        if not ok:
            continue
        long_n += 1
        long_hits += int(actual == 1)

    acc = long_hits / long_n if long_n else 0.0
    return {
        "filtered_accuracy": round(acc * 100, 2),
        "n_signals": long_n,
        "n_long": long_n,
        "long_hit_rate": round(acc * 100, 2) if long_n else None,
    }


def optimize_thresholds_v5(
    cal_df: pd.DataFrame,
    min_samples: int = 8,
    long_only: bool = True,
) -> dict:
    if cal_df.empty:
        return {
            "th_up": 0.57,
            "th_down": 0.44,
            "min_agree": 4,
            "filtered_accuracy": None,
            "n_signals": 0,
            "quality_filter": {"max_rsi14": 76, "max_price_pos20": 0.92},
        }

    best_score = -1.0
    best = {
        "th_up": 0.57,
        "th_down": 0.44,
        "min_agree": 4,
        "filtered_accuracy": 0.0,
        "n_signals": 0,
        "quality_filter": {"max_rsi14": 76, "max_price_pos20": 0.92},
    }
    quality_opts = [
        {"max_rsi14": 76, "max_price_pos20": 0.92},
        {"max_rsi14": 74, "max_price_pos20": 0.90},
        {"max_rsi14": 72, "max_price_pos20": 0.88},
        {"max_rsi14": 78, "max_price_pos20": 0.94, "min_ret5": -0.08},
    ]

    for th_up in np.arange(0.52, 0.66, 0.01):
        for min_agree in (3, 4, 5):
            for qcfg in quality_opts:
                fd = eval_filter_v5(cal_df, th_up, min_agree, qcfg, long_only=long_only)
                n = fd["n_long"]
                if n < min_samples:
                    continue
                acc = fd["filtered_accuracy"] / 100.0
                score = acc * (n ** 0.35)
                if acc >= 0.58:
                    score *= 1.05
                if acc >= 0.60:
                    score *= 1.10
                if score > best_score:
                    best_score = score
                    best = {
                        "th_up": round(float(th_up), 2),
                        "th_down": 0.44,
                        "min_agree": min_agree,
                        "filtered_accuracy": fd["filtered_accuracy"],
                        "n_signals": n,
                        "n_long": n,
                        "long_hit_rate": fd["long_hit_rate"],
                        "quality_filter": qcfg,
                    }
    return best


def optimize_and_save_thresholds(cal1: pd.DataFrame, cal5: pd.DataFrame, long_only: bool = True) -> dict:
    best1 = {"th_up": 0.62, "th_down": 0.38, "min_agree": 3}
    best5 = optimize_thresholds_v5(cal5, min_samples=6, long_only=long_only)
    return {
        "version": 5,
        "long_only": long_only,
        "filter": {
            "1d": {k: best1[k] for k in ("th_up", "th_down", "min_agree")},
            "5d": {k: best5[k] for k in ("th_up", "th_down", "min_agree")},
        },
        "quality_filter": best5.get("quality_filter", {"max_rsi14": 76, "max_price_pos20": 0.92}),
        "calibration_stats": {
            "1d": {"note": "日度默认观望，不参与阈值优化"},
            "5d": {k: best5[k] for k in ("filtered_accuracy", "n_signals", "long_hit_rate") if k in best5},
        },
    }


def format_calibration_report(m1: dict, m5: dict) -> str:
    lines = [
        "=" * 52,
        "  原油(SC) 预测校准 v5（WTI + 美库存 + PMI + 仅做多）",
        "=" * 52,
        "",
        "  【下一交易日】— 建议一律观望",
        f"  全样本准确率     : {m1.get('accuracy', 0)}%  (n={m1.get('samples', 0)})",
        "",
        "  【未来一周 5日 — 仅做多信号】",
        f"  全样本准确率     : {m5.get('accuracy', 0)}%  (n={m5.get('samples', 0)})",
        f"  做多信号准确率   : {m5.get('filtered_accuracy', 'N/A')}%  (信号数={m5.get('filtered_n_signals', 0)})",
        f"  做多命中率       : {m5.get('filtered_long_hit', 'N/A')}%  (n={m5.get('filtered_n_long', 0)})",
        f"  Brier分数        : {m5.get('brier_score', 0)}",
        "",
        "=" * 52,
    ]
    return "\n".join(lines)
