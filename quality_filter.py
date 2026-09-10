"""信号质量过滤 — 避免超买追多等低质量做多。"""
from __future__ import annotations

import pandas as pd


DEFAULT_QUALITY = {
    "max_rsi14": 76.0,
    "max_price_pos20": 0.92,
    "min_ret5": -0.08,  # 近5日跌超8%时不追多
}


def passes_quality(row: pd.Series, cfg: dict | None = None) -> tuple[bool, str]:
    """5日做多前的质量检查。"""
    q = {**DEFAULT_QUALITY, **(cfg or {})}
    rsi = row.get("rsi14")
    if pd.notna(rsi) and float(rsi) > q["max_rsi14"]:
        return False, f"RSI={float(rsi):.1f}>{q['max_rsi14']:.0f}超买区"
    pos = row.get("price_pos20")
    if pd.notna(pos) and float(pos) > q["max_price_pos20"]:
        return False, f"20日位置{float(pos)*100:.0f}%>{q['max_price_pos20']*100:.0f}%高位"
    ret5 = row.get("ret5")
    if pd.notna(ret5) and float(ret5) < q["min_ret5"]:
        return False, f"近5日跌{float(ret5)*100:.1f}%过深"
    return True, ""


def apply_quality_to_signal(
    signal: str,
    reason: str,
    row: pd.Series,
    cfg: dict | None = None,
) -> tuple[str, str]:
    if signal != "做多":
        return signal, reason
    ok, why = passes_quality(row, cfg)
    if ok:
        return signal, reason
    return "观望", f"质量过滤：{why}"
