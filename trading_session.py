"""分时段预测目标 — 14:15 夜盘 / 20:50 明日日盘。"""
from __future__ import annotations

from datetime import datetime

from trading_labels import _WEEKDAY, _parse_date, is_weekend, next_trading_day, calendar_tomorrow

SLOT_NIGHT = "1415"
SLOT_DAY = "2050"


def resolve_session_mode(slot: str) -> str:
    """daily | night | day"""
    slot = (slot or "").replace(":", "")
    if slot == SLOT_NIGHT:
        return "night"
    if slot == SLOT_DAY:
        return "day"
    return "daily"


def format_night_session_label(as_of_date: str) -> tuple[str, str | None]:
    """
    14:15 推送：预测今夜 21:00 起的夜盘。
    国内规则：该夜盘计入「下一交易日」的 K 线（非日历明日）。
    """
    as_of = _parse_date(as_of_date)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ref = max(as_of, today)
    trade_day = next_trading_day(ref)
    md = trade_day.strftime("%m/%d")
    wd = _WEEKDAY[trade_day.weekday()]
    label = f"今夜夜盘(计入{wd}{md}交易日)"
    note = None
    if ref.weekday() == 4:
        note = "周五夜盘归属下周一交易日，非日历周六"
    elif is_weekend(calendar_tomorrow(ref)):
        cal = calendar_tomorrow(ref)
        note = f"日历明日{_WEEKDAY[cal.weekday()]}休市，夜盘指{wd}{md}交易日"
    return label, note


def format_day_session_label(as_of_date: str) -> tuple[str, str | None]:
    """
    20:50 推送：预测下一交易日日盘 9:00–15:00。
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target = next_trading_day(today)
    md = target.strftime("%m/%d")
    wd = _WEEKDAY[target.weekday()]
    label = f"明日日盘({wd}{md})"
    note = None
    if today.weekday() == 4 or is_weekend(calendar_tomorrow(today)):
        note = "周末休市，日盘指下一交易日白天时段"
    return label, note
