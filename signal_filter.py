"""弱信号过滤与子模型一致性检验。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _agreement_keys(horizon: int, extra_5d_keys: list[str] | None = None) -> list[str]:
    if horizon == 1:
        return ["sim1", "log1", "mom1", "fund"]
    keys = ["sim5", "log5", "mc5", "fund", "lme"]
    if extra_5d_keys:
        keys.extend(extra_5d_keys)
    return keys


def count_agreement(
    probs: dict,
    horizon: int = 1,
    extra_5d_keys: list[str] | None = None,
) -> tuple[int, int]:
    """返回 (看多票数, 看空票数)。"""
    keys = _agreement_keys(horizon, extra_5d_keys)

    bullish = bearish = 0
    for k in keys:
        v = probs.get(k)
        if v is None:
            continue
        if v >= 0.52:
            bullish += 1
        elif v <= 0.48:
            bearish += 1
    return bullish, bearish


def classify_signal(
    prob_up: float,
    probs: dict,
    horizon: int,
    th_up: float,
    th_down: float,
    min_agree: int = 3,
    long_only: bool = False,
    extra_5d_keys: list[str] | None = None,
) -> tuple[str, str]:
    """
    返回 (信号, 说明)
    信号: 做多 / 做空 / 观望
    """
    bull, bear = count_agreement(probs, horizon, extra_5d_keys=extra_5d_keys)
    max_agree = len(_agreement_keys(horizon, extra_5d_keys))

    if prob_up >= th_up and bull >= min_agree:
        return "做多", f"概率{prob_up*100:.1f}%≥{th_up*100:.0f}%, {bull}/{max_agree}项子模型看多"
    if not long_only and prob_up <= th_down and bear >= min_agree:
        return "做空", f"概率{prob_up*100:.1f}%≤{th_down*100:.0f}%, {bear}/{max_agree}项子模型看空"
    if long_only and prob_up <= th_down and bear >= min_agree:
        return "观望", f"仅做多模式：原偏空信号已过滤({prob_up*100:.1f}%)"
    if th_down < prob_up < th_up:
        return "观望", f"概率{prob_up*100:.1f}%处于弱信号区({th_down*100:.0f}%~{th_up*100:.0f}%)"
    if prob_up >= th_up:
        return "观望", f"概率偏多但子模型一致性不足({bull}/{min_agree})"
    if prob_up <= th_down:
        return "观望", f"概率偏空但子模型一致性不足({bear}/{min_agree})"
    return "观望", "信号不明确"


def optimize_thresholds(
    cal_df: pd.DataFrame,
    min_samples: int = 15,
    long_only: bool = False,
) -> dict:
    """在历史校准集上搜索最优阈值。long_only 时只优化做多阈值。"""
    if cal_df.empty:
        return {"th_up": 0.56, "th_down": 0.44, "min_agree": 3, "filtered_accuracy": None, "n_signals": 0}

    best_score = -1.0
    best = {"th_up": 0.56, "th_down": 0.44, "min_agree": 3, "filtered_accuracy": 0.0, "n_signals": 0}

    th_down_range = [0.44] if long_only else list(np.arange(0.34, 0.48, 0.01))
    for th_up in np.arange(0.52, 0.66, 0.01):
        for th_down in th_down_range:
            if not long_only and th_down >= th_up - 0.04:
                continue
            for min_agree in (3,):
                stats = _eval_filter(cal_df, th_up, th_down, min_agree, long_only=long_only)
                if stats["n_long"] < min_samples:
                    continue
                score = stats["accuracy"] * np.sqrt(stats["n_long"])
                if score > best_score:
                    best_score = score
                    best = {
                        "th_up": round(float(th_up), 2),
                        "th_down": round(float(th_down), 2),
                        "min_agree": min_agree,
                        "filtered_accuracy": round(stats["accuracy"] * 100, 2),
                        "n_signals": stats["n_long"],
                        "n_long": stats["n_long"],
                        "n_short": 0 if long_only else stats["n_short"],
                        "long_hit_rate": round(stats["long_hit_rate"] * 100, 2) if stats.get("long_hit_rate") else None,
                    }
    return best


def _eval_filter(
    cal_df: pd.DataFrame,
    th_up: float,
    th_down: float,
    min_agree: int,
    long_only: bool = False,
    extra_5d_keys: list[str] | None = None,
) -> dict:
    long_hits = long_n = short_hits = short_n = 0
    for _, r in cal_df.iterrows():
        probs = r.get("probs_dict") or {}
        bull, bear = count_agreement(
            probs, int(r.get("horizon", 1)), extra_5d_keys=extra_5d_keys if int(r.get("horizon", 1)) == 5 else None
        )
        p, actual = r["pred_prob_up"], r["actual_up"]
        if p >= th_up and bull >= min_agree:
            long_n += 1
            long_hits += int(actual == 1)
        elif not long_only and p <= th_down and bear >= min_agree:
            short_n += 1
            short_hits += int(actual == 0)

    if long_only:
        if long_n == 0:
            return {"accuracy": 0.0, "n_signals": 0, "n_long": 0, "n_short": 0, "long_hit_rate": 0.0}
        return {
            "accuracy": long_hits / long_n,
            "n_signals": long_n,
            "n_long": long_n,
            "n_short": 0,
            "long_hit_rate": long_hits / long_n,
        }

    total = long_n + short_n
    if total == 0:
        return {"accuracy": 0.0, "n_signals": 0, "n_long": 0, "n_short": 0, "long_hit_rate": 0.0}
    return {
        "accuracy": (long_hits + short_hits) / total,
        "n_signals": total,
        "n_long": long_n,
        "n_short": short_n,
        "long_hit_rate": long_hits / long_n if long_n else 0.0,
    }


def eval_filter_detailed(
    cal_df: pd.DataFrame,
    th_up: float,
    th_down: float,
    min_agree: int,
    long_only: bool = False,
    extra_5d_keys: list[str] | None = None,
) -> dict:
    """详细评估过滤后准确率。"""
    long_hits = long_n = short_hits = short_n = 0
    for _, r in cal_df.iterrows():
        probs = r.get("probs_dict", {})
        bull, bear = count_agreement(
            probs, int(r["horizon"]), extra_5d_keys=extra_5d_keys if int(r["horizon"]) == 5 else None
        )
        p, actual = r["pred_prob_up"], r["actual_up"]
        if p >= th_up and bull >= min_agree:
            long_n += 1
            long_hits += int(actual == 1)
        elif not long_only and p <= th_down and bear >= min_agree:
            short_n += 1
            short_hits += int(actual == 0)

    if long_only:
        acc = long_hits / long_n if long_n else 0.0
        total = long_n
    else:
        total = long_n + short_n
        acc = (long_hits + short_hits) / total if total else 0.0

    return {
        "filtered_accuracy": round(acc * 100, 2),
        "n_signals": total,
        "n_long": long_n,
        "n_short": short_n,
        "long_hit_rate": round(long_hits / long_n * 100, 2) if long_n else None,
        "short_hit_rate": round(short_hits / short_n * 100, 2) if short_n and not long_only else None,
    }
