"""交易日文案 — 周末/下交易日标注（用于推送）。"""
from __future__ import annotations

from datetime import datetime, timedelta

_WEEKDAY = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _parse_date(s: str) -> datetime:
    s = str(s).replace("-", "")[:8]
    return datetime.strptime(s, "%Y%m%d")


def next_trading_day(after: datetime) -> datetime:
    """after 之后第一个周一至周五。"""
    d = after
    while True:
        d += timedelta(days=1)
        if d.weekday() < 5:
            return d


def calendar_tomorrow(ref: datetime) -> datetime:
    return ref + timedelta(days=1)


def is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5


def format_1d_horizon_label(as_of_date: str) -> tuple[str, str | None]:
    """
    根据数据截至日 as_of，生成 1 日预测行的前缀与可选备注。
    若日历次日为周六/周日，明确标注下交易日与周末休市。
    """
    as_of = _parse_date(as_of_date)
    cal_next = calendar_tomorrow(as_of)
    trade_next = next_trading_day(as_of)
    md = trade_next.strftime("%m/%d")
    wd = _WEEKDAY[trade_next.weekday()]

    if is_weekend(cal_next):
        cal_md = cal_next.strftime("%m/%d")
        cal_wd = _WEEKDAY[cal_next.weekday()]
        label = f"下交易日({wd}{md})"
        note = f"明天{cal_md}({cal_wd})休市，1日指下交易日{md}({wd})"
        return label, note
    label = f"明日({wd}{md})"
    return label, None


def format_5d_horizon_label(as_of_date: str) -> tuple[str, str | None]:
    """5 日预测行前缀；若跨周末则注明含休市。"""
    as_of = _parse_date(as_of_date)
    cal_next = calendar_tomorrow(as_of)
    end = as_of
    count = 0
    while count < 5:
        end += timedelta(days=1)
        if end.weekday() < 5:
            count += 1
    md = end.strftime("%m/%d")
    wd = _WEEKDAY[end.weekday()]
    label = f"一周至({wd}{md})"
    note = None
    if is_weekend(cal_next) or as_of.weekday() == 4:
        note = "5个交易日，含周末休市"
    return label, note


def format_push_header(as_of_date: str, slot: str, pushed_at: datetime | None = None) -> str:
    """推送头：当前推送时刻 + 数据截至日。"""
    pushed_at = pushed_at or datetime.now()
    slot_label = f"{slot[:2]}:{slot[2:]}" if len(slot) == 4 else slot
    as_of = _parse_date(as_of_date)
    return f"推送 {pushed_at.strftime('%m-%d %H:%M')} 时段{slot_label} | 数据截至 {as_of.strftime('%Y-%m-%d')}"
