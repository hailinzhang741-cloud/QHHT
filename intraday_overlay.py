"""盘中概率微调层 — 不修改 v5 日 K 核心，仅叠加 1 日概率。"""
from __future__ import annotations

from dataclasses import replace

from forecast import ForecastResult
from intraday_snapshot import IntradayBundle, IntradaySnapshot


def _clamp(p: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, p))


def overlay_delta_1d(snap: IntradaySnapshot | None, wti: IntradaySnapshot | None = None) -> float:
    """公开：由盘中快照计算 1 日概率修正量（供回测使用）。"""
    return _delta_from_snapshot(snap, wti)


def adjust_prob_up_1d(base_prob: float, snap: IntradaySnapshot | None, wti: IntradaySnapshot | None = None) -> float:
    if snap is None:
        return base_prob
    return _clamp(base_prob + overlay_delta_1d(snap, wti))


def _delta_from_snapshot(snap: IntradaySnapshot | None, wti: IntradaySnapshot | None = None) -> float:
    if snap is None:
        return 0.0
    delta = 0.0
    r = snap.session_ret_open_pct / 100.0
    if r > 0.012:
        delta += 0.04
    elif r > 0.004:
        delta += 0.02
    elif r < -0.012:
        delta -= 0.04
    elif r < -0.004:
        delta -= 0.02
    rs = snap.session_ret_settle_pct / 100.0
    if rs > 0.008:
        delta += 0.015
    elif rs < -0.008:
        delta -= 0.015
    if wti is not None and wti.session_ret_settle_pct:
        wt = wti.session_ret_settle_pct / 100.0
        delta += max(-0.02, min(0.02, wt * 0.4))
    return max(-0.06, min(0.06, delta))


def apply_intraday_overlay(result: ForecastResult, snap: IntradaySnapshot | None, wti: IntradaySnapshot | None = None) -> ForecastResult:
    if snap is None:
        return result
    delta = _delta_from_snapshot(snap, wti=wti if snap.code == "SC0" else None)
    p1 = _clamp(result.prob_up_1d + delta)
    sig = dict(result.signals)
    sig["盘中_1日修正"] = round(delta * 100, 2)
    sig["盘中_来源"] = snap.source
    sig["盘中_今开涨跌_%"] = snap.session_ret_open_pct
    sig["盘中_结算涨跌_%"] = snap.session_ret_settle_pct
    return replace(
        result,
        prob_up_1d=round(p1, 4),
        prob_down_1d=round(1 - p1, 4),
        signals=sig,
    )


def apply_bundle(cu_result: ForecastResult, oil_result: ForecastResult, bundle: IntradayBundle) -> tuple[ForecastResult, ForecastResult, IntradayBundle]:
    cu_out = apply_intraday_overlay(cu_result, bundle.cu)
    oil_out = apply_intraday_overlay(oil_result, bundle.oil, wti=bundle.wti)
    return cu_out, oil_out, bundle
