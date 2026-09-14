"""分时段概率 — 14:15 夜盘 / 20:50 明日日盘（基于 v5 日 K + 盘中快照轻量融合）。"""
from __future__ import annotations

from dataclasses import dataclass

from config import load_env
from intraday_snapshot import IntradayBundle, IntradaySnapshot
from trading_session import (
    SLOT_DAY,
    SLOT_NIGHT,
    format_day_session_label,
    format_night_session_label,
    resolve_session_mode,
)


def session_forecast_enabled() -> bool:
    return load_env("SESSION_FORECAST", "1").strip().lower() in ("1", "true", "yes")


def _clamp(p: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, p))


def _session_signal(prob_up: float) -> str:
    if prob_up >= 0.62:
        return "做多"
    if prob_up <= 0.38:
        return "做空"
    return "观望"


def _prob_night_session(base_prob_1d: float, snap: IntradaySnapshot | None, wti: IntradaySnapshot | None, is_oil: bool) -> float:
    """14:15 夜盘：日 K 1 日概率 + 日盘强弱（延续/反转轻量修正）。"""
    p = base_prob_1d
    if snap:
        r = snap.session_ret_open_pct / 100.0
        if r > 0.010:
            p += 0.025
        elif r > 0.004:
            p += 0.012
        elif r < -0.010:
            p -= 0.025
        elif r < -0.004:
            p -= 0.012
    if is_oil and wti:
        wt = wti.session_ret_settle_pct / 100.0
        p += max(-0.02, min(0.02, wt * 0.35))
    return _clamp(p)


def _prob_day_session(base_prob_1d: float, snap: IntradaySnapshot | None, wti: IntradaySnapshot | None, is_oil: bool) -> float:
    """20:50 明日日盘：以 v5 下一日 K 为主（≈下一日盘方向），叠加夜前 WTI/当日信息。"""
    p = base_prob_1d
    if is_oil and wti:
        wt = wti.session_ret_settle_pct / 100.0
        p += max(-0.025, min(0.025, wt * 0.45))
    if snap:
        rs = snap.session_ret_settle_pct / 100.0
        p += max(-0.015, min(0.015, rs * 0.25))
    return _clamp(p)


@dataclass
class SessionOutlook:
    mode: str
    label: str
    note: str | None
    prob_up: float
    prob_down: float
    signal: str


def build_session_outlook(
    result,
    slot: str,
    snap: IntradaySnapshot | None = None,
    wti: IntradaySnapshot | None = None,
    is_oil: bool = False,
) -> SessionOutlook | None:
    if not session_forecast_enabled():
        return None
    mode = resolve_session_mode(slot)
    if mode == "daily":
        return None

    base = float(result.prob_up_1d)
    if mode == "night":
        label, note = format_night_session_label(result.as_of_date)
        p = _prob_night_session(base, snap, wti, is_oil)
    else:
        label, note = format_day_session_label(result.as_of_date)
        p = _prob_day_session(base, snap, wti, is_oil)

    return SessionOutlook(
        mode=mode,
        label=label,
        note=note,
        prob_up=round(p, 4),
        prob_down=round(1 - p, 4),
        signal=_session_signal(p),
    )


def build_session_bundle(
    cu_result,
    oil_result,
    slot: str,
    intraday: IntradayBundle | None,
) -> tuple[SessionOutlook | None, SessionOutlook | None]:
    snap_cu = intraday.cu if intraday else None
    snap_oil = intraday.oil if intraday else None
    wti = intraday.wti if intraday else None
    cu = build_session_outlook(cu_result, slot, snap_cu, wti, is_oil=False)
    oil = build_session_outlook(oil_result, slot, snap_oil, wti, is_oil=True)
    return cu, oil
