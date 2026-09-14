"""盘中修正规则回测 — 对比纯 v5 与叠加盘中修正后的 1 日准确率。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_overlay import adjust_prob_up_1d
from intraday_snapshot import IntradaySnapshot

SLOTS = ("0840", "1030", "1415", "2050")
# 用日 K 近似各推送时刻价格（无分钟权限时的 walk-forward 代理）
SLOT_FRAC = {"0840": 0.0, "1030": 0.35, "1415": 0.75, "2050": 1.0}


def simulate_snapshot_from_daily(
    row: pd.Series,
    prev_settle: float,
    code: str,
    slot: str,
) -> IntradaySnapshot:
    open_ = float(row["open"])
    close = float(row["close"])
    frac = SLOT_FRAC.get(slot, 1.0)
    last = open_ if slot == "0840" else open_ + frac * (close - open_)
    tdate = str(row["trade_date"])
    return IntradaySnapshot(
        code=code,
        trade_date=tdate,
        quote_time=slot,
        open=open_,
        high=float(row.get("high") or max(open_, close)),
        low=float(row.get("low") or min(open_, close)),
        last=last,
        prev_close=prev_settle,
        prev_settle=prev_settle,
        volume=float(row.get("vol") or row.get("volume") or 0),
        open_interest=float(row.get("oi") or 0),
        session_ret_open_pct=round((last - open_) / open_ * 100, 3) if open_ else 0.0,
        session_ret_settle_pct=round((last - prev_settle) / prev_settle * 100, 3) if prev_settle else 0.0,
        source=f"backtest_proxy_{slot}",
    )


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    eps = 1e-9
    pred_label = (pred >= 0.5).astype(int)
    brier = float(np.mean((pred - actual) ** 2))
    logloss = float(-np.mean(actual * np.log(pred + eps) + (1 - actual) * np.log(1 - pred + eps)))
    return {
        "samples": int(len(pred)),
        "accuracy": round(float((pred_label == actual).mean()) * 100, 2),
        "brier": round(brier, 4),
        "log_loss": round(logloss, 4),
    }


def run_intraday_backtest(
    feat: pd.DataFrame,
    compute_probs_fn,
    code: str,
    min_train: int = 252,
    step: int = 3,
    slots: tuple[str, ...] = SLOTS,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    closes = feat["close"].astype(float).values

    for idx in range(min_train, len(feat) - 1, step):
        row = feat.iloc[idx]
        if pd.isna(row.get("next_up1")):
            continue
        prev_settle = float(closes[idx - 1]) if idx > 0 else float(row["open"])
        probs = compute_probs_fn(feat, idx, use_fundamentals=True)
        base_p = float(probs["prob_up_1d"])
        actual = int(row["next_up1"])

        for slot in slots:
            snap = simulate_snapshot_from_daily(row, prev_settle, code, slot)
            adj_p = adjust_prob_up_1d(base_p, snap)
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "slot": slot,
                    "base_prob": base_p,
                    "adj_prob": adj_p,
                    "delta": adj_p - base_p,
                    "actual_up": actual,
                    "base_correct": int((base_p >= 0.5) == actual),
                    "adj_correct": int((adj_p >= 0.5) == actual),
                    "session_ret_open_pct": snap.session_ret_open_pct,
                }
            )

    df = pd.DataFrame(rows)
    summary: dict = {"by_slot": {}, "all_slots": {}}
    if df.empty:
        return df, summary

    for slot in slots:
        sub = df[df["slot"] == slot]
        if sub.empty:
            continue
        base_m = _metrics(sub["base_prob"].values, sub["actual_up"].values)
        adj_m = _metrics(sub["adj_prob"].values, sub["actual_up"].values)
        improved = sub["adj_correct"].sum() - sub["base_correct"].sum()
        summary["by_slot"][slot] = {
            "base": base_m,
            "adj": adj_m,
            "accuracy_delta_pp": round(adj_m["accuracy"] - base_m["accuracy"], 2),
            "brier_delta": round(adj_m["brier"] - base_m["brier"], 4),
            "logloss_delta": round(adj_m["log_loss"] - base_m["log_loss"], 4),
            "n_improved_samples": int(improved),
        }

    base_all = _metrics(df["base_prob"].values, df["actual_up"].values)
    adj_all = _metrics(df["adj_prob"].values, df["actual_up"].values)
    summary["all_slots"] = {
        "base": base_all,
        "adj": adj_all,
        "accuracy_delta_pp": round(adj_all["accuracy"] - base_all["accuracy"], 2),
        "brier_delta": round(adj_all["brier"] - base_all["brier"], 4),
        "logloss_delta": round(adj_all["log_loss"] - base_all["log_loss"], 4),
    }
    return df, summary


def format_report(name: str, summary: dict) -> str:
    lines = [
        "=" * 56,
        f"  {name} — 1日概率 纯v5 vs 盘中修正 回测对比",
        "=" * 56,
        "  说明: 用日K OHLC 近似 08:40/10:30/14:15/20:50 盘中价（代理回测）",
        "",
    ]
    all_s = summary.get("all_slots") or {}
    if all_s:
        b, a = all_s["base"], all_s["adj"]
        lines += [
            "  【四时段合并】",
            f"  纯v5  准确率 {b['accuracy']}%  Brier {b['brier']}  LogLoss {b['log_loss']}  (n={b['samples']})",
            f"  修正后 准确率 {a['accuracy']}%  Brier {a['brier']}  LogLoss {a['log_loss']}",
            f"  变化    准确率 {all_s['accuracy_delta_pp']:+.2f}pp  Brier {all_s['brier_delta']:+.4f}  LogLoss {all_s['logloss_delta']:+.4f}",
            "",
        ]
    lines.append("  【分时段】")
    for slot, s in (summary.get("by_slot") or {}).items():
        b, a = s["base"], s["adj"]
        t = f"{slot[:2]}:{slot[2:]}"
        lines.append(
            f"  {t}  纯v5 {b['accuracy']}% -> 修正 {a['accuracy']}%  ({s['accuracy_delta_pp']:+.2f}pp)  "
            f"Brier {s['brier_delta']:+.4f}  样本改善 {s['n_improved_samples']:+d}"
        )
    lines.append("")
    lines.append("  Brier/LogLoss 下降 = 概率校准更好；准确率 ±0~2pp 内通常视为持平。")
    lines.append("=" * 56)
    return "\n".join(lines)
