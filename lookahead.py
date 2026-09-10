"""未来函数审计与防护工具。"""
from __future__ import annotations

import pandas as pd


def split_known_outcomes(
    usable: pd.DataFrame,
    horizon: int,
    end_idx: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    在 end_idx 时点，仅保留 label 已知的 history 行。
    row j 的 next_up_h 需要 close[j+h]，当 j+h <= end_idx 时为已知。
    """
    if end_idx is not None:
        usable = usable.iloc[: end_idx + 1]
    if len(usable) <= horizon:
        return usable.iloc[-1], usable.iloc[:0]
    latest = usable.iloc[-1]
    hist = usable.iloc[: len(usable) - horizon]
    return latest, hist


def prev_month(ym: str) -> str:
    ts = pd.Timestamp(f"{ym}01") - pd.DateOffset(months=1)
    return ts.strftime("%Y%m")


def audit_report_v4_v5() -> str:
    """静态审计清单（代码层）。"""
    lines = [
        "=" * 52,
        "  未来函数审计 v4 / v5",
        "=" * 52,
        "",
        "【已修复】",
        "  1. 相似形态 / 逻辑回归：history 仅含 j+h<=T 的样本",
        "  2. PMI：trade_date 所在月 M 仅使用 M-1 月 PMI（发布滞后）",
        "",
        "【无问题】",
        "  - 技术指标：rolling/pct_change 仅向后看",
        "  - 蒙特卡洛：仅用历史 ret1",
        "  - 仓单/持仓/沪伦比/LME库存：按 trade_date 当日或 ffill",
        "  - 校准回测：compute_probs(feat, end_idx) 截断至 T",
        "",
        "【实盘注意】",
        "  - 08:30 跑批：前一日仓单/持仓/LME 已公布，无未来数据",
        "  - 沪伦比同日对齐：LME 与沪铜均为已收盘数据",
        "  - PMI 每月初更新，月中不变",
        "=" * 52,
    ]
    return "\n".join(lines)
