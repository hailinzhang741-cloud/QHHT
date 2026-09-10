"""概率校准回测 v4 — 沪伦比 + 仅做多 + 过滤后准确率。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import compute_probs, prepare_features
from signal_filter import eval_filter_detailed, optimize_thresholds


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
            }
        )

    return pd.DataFrame(rows)


def calibration_metrics(cal_df: pd.DataFrame, filter_cfg: dict | None = None, long_only: bool = False) -> dict:
    if cal_df.empty:
        return {}

    p = cal_df["pred_prob_up"].values
    y = cal_df["actual_up"].values
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-9
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    result = {
        "samples": len(cal_df),
        "accuracy": round(float(cal_df["correct"].mean()) * 100, 2),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "baseline_up_rate": round(float(y.mean()) * 100, 2),
    }

    if filter_cfg:
        fd = eval_filter_detailed(
            cal_df,
            filter_cfg.get("th_up", 0.54),
            filter_cfg.get("th_down", 0.45),
            filter_cfg.get("min_agree", 3),
            long_only=long_only,
        )
        result.update(
            {
                "filtered_accuracy": fd["filtered_accuracy"],
                "filtered_n_signals": fd["n_signals"],
                "filtered_n_long": fd["n_long"],
                "filtered_n_short": fd["n_short"],
                "filtered_long_hit": fd["long_hit_rate"],
                "filtered_short_hit": fd["short_hit_rate"],
            }
        )
    return result


def optimize_and_save_thresholds(cal1: pd.DataFrame, cal5: pd.DataFrame, long_only: bool = True) -> dict:
    best1 = {"th_up": 0.62, "th_down": 0.38, "min_agree": 3}
    best5 = optimize_thresholds(cal5, min_samples=12, long_only=long_only)
    return {
        "version": 4,
        "long_only": long_only,
        "filter": {
            "1d": {k: best1[k] for k in ("th_up", "th_down", "min_agree")},
            "5d": {k: best5[k] for k in ("th_up", "th_down", "min_agree")},
        },
        "calibration_stats": {
            "1d": {"note": "日度默认观望，不参与阈值优化"},
            "5d": {k: best5[k] for k in ("filtered_accuracy", "n_signals", "long_hit_rate") if k in best5},
        },
    }


def format_calibration_report(m1: dict, m5: dict) -> str:
    lines = [
        "=" * 52,
        "  沪铜预测校准 v4（沪伦比 + 仅做多）",
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
        "  v4: 仅输出「做多」信号；做空已禁用；有信号时推送通知。",
        "=" * 52,
    ]
    return "\n".join(lines)
