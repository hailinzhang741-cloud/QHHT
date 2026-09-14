"""盘中实时快照 — 多数据源链（新浪主连/spot 优先，akshare 备用）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import akshare as ak
import pandas as pd

from config import load_env

# 数据源优先级说明（Phase 1 已实现）:
# 1. akshare futures_zh_spot(CU0/SC0) — 国内期货 quasi-realtime（约 3~15 秒延迟）
# 2. akshare futures_main_sina — 主连当日 OHLC 备用
# 3. akshare futures_foreign_commodity_realtime(CL) — WTI 外盘实时
# 未接入（需更高权限/付费）:
# - Tushare ft_mins / rt_fut_min（当前 token 无权限）
# - CTP/券商 Level-1 tick


@dataclass
class IntradaySnapshot:
    code: str
    trade_date: str
    quote_time: str
    open: float
    high: float
    low: float
    last: float
    prev_close: float
    prev_settle: float
    volume: float
    open_interest: float
    session_ret_open_pct: float
    session_ret_settle_pct: float
    source: str
    extra: dict = field(default_factory=dict)

    def format_line(self, unit: str = "元/吨") -> str:
        t = self.quote_time
        if len(t) == 6 and t.isdigit():
            t = f"{t[:2]}:{t[2:4]}:{t[4:6]}"
        return (
            f"[盘中 {t}] 现价 {self.last:,.1f}{unit}  "
            f"今开→现价 {self.session_ret_open_pct:+.2f}%  "
            f"结算→现价 {self.session_ret_settle_pct:+.2f}%  "
            f"量 {int(self.volume):,}  持仓 {int(self.open_interest):,}  ({self.source})"
        )


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _pct(n: float, d: float) -> float:
    if not d:
        return 0.0
    return (n - d) / d * 100.0


def _from_spot_row(code: str, row: pd.Series) -> IntradaySnapshot:
    last = float(row.get("current_price") or row.get("last") or 0)
    open_ = float(row.get("open") or 0)
    settle = float(row.get("last_settle_price") or open_ or last)
    prev_close = float(row.get("last_close") or settle)
    tdate = _today_str()
    return IntradaySnapshot(
        code=code,
        trade_date=tdate,
        quote_time=str(row.get("time") or ""),
        open=open_,
        high=float(row.get("high") or last),
        low=float(row.get("low") or last),
        last=last,
        prev_close=prev_close,
        prev_settle=settle,
        volume=float(row.get("volume") or 0),
        open_interest=float(row.get("hold") or 0),
        session_ret_open_pct=round(_pct(last, open_), 3),
        session_ret_settle_pct=round(_pct(last, settle), 3),
        source="akshare_spot",
        extra={"bid": row.get("bid_price"), "ask": row.get("ask_price"), "avg": row.get("avg_price")},
    )


def fetch_akshare_spot(main_code: str) -> IntradaySnapshot | None:
    """main_code: CU0 / SC0"""
    try:
        df = ak.futures_zh_spot(symbol=main_code, market="CF", adjust="0")
        if df is None or df.empty:
            return None
        return _from_spot_row(main_code, df.iloc[0])
    except Exception as exc:
        print(f"[intraday] akshare_spot {main_code} 失败: {exc}")
        return None


def fetch_akshare_main_sina(main_code: str) -> IntradaySnapshot | None:
    try:
        df = ak.futures_main_sina(symbol=main_code)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        # columns: 日期 开盘价 最高价 最低价 收盘价 成交量 持仓量 动态结算价
        date_val = row.iloc[0]
        if hasattr(date_val, "strftime"):
            tdate = date_val.strftime("%Y%m%d")
        else:
            tdate = str(date_val).replace("-", "")[:8]
        if tdate != _today_str():
            print(f"[intraday] main_sina {main_code} 末行日期 {tdate} 非今日，跳过")
            return None
        open_ = float(row.iloc[1])
        high = float(row.iloc[2])
        low = float(row.iloc[3])
        close = float(row.iloc[4])
        vol = float(row.iloc[5])
        oi = float(row.iloc[6])
        settle = float(row.iloc[7]) if len(row) > 7 and pd.notna(row.iloc[7]) else close
        return IntradaySnapshot(
            code=main_code,
            trade_date=tdate,
            quote_time=datetime.now().strftime("%H%M%S"),
            open=open_,
            high=high,
            low=low,
            last=close,
            prev_close=close,
            prev_settle=settle,
            volume=vol,
            open_interest=oi,
            session_ret_open_pct=round(_pct(close, open_), 3),
            session_ret_settle_pct=round(_pct(close, settle), 3),
            source="akshare_main_sina",
        )
    except Exception as exc:
        print(f"[intraday] main_sina {main_code} 失败: {exc}")
        return None


def fetch_wti_realtime() -> IntradaySnapshot | None:
    try:
        df = ak.futures_foreign_commodity_realtime(symbol="CL")
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        last = float(row.get("最新价") or row.iloc[1])
        open_ = float(row.get("开盘价") or row.iloc[5])
        high = float(row.get("最高价") or row.iloc[6])
        low = float(row.get("最低价") or row.iloc[7])
        prev = float(row.get("昨日结算") or row.iloc[8])
        tdate = str(row.get("日期") or _today_str()).replace("-", "")[:8]
        qtime = str(row.get("更新时间") or datetime.now().strftime("%H:%M:%S"))
        return IntradaySnapshot(
            code="CL",
            trade_date=tdate,
            quote_time=qtime.replace(":", ""),
            open=open_,
            high=high,
            low=low,
            last=last,
            prev_close=prev,
            prev_settle=prev,
            volume=0.0,
            open_interest=0.0,
            session_ret_open_pct=round(_pct(last, open_), 3),
            session_ret_settle_pct=round(_pct(last, prev), 3),
            source="akshare_wti_realtime",
            extra={"chg_pct": row.get("涨跌幅")},
        )
    except Exception as exc:
        print(f"[intraday] WTI realtime 失败: {exc}")
        return None


def fetch_snapshot(main_code: str) -> IntradaySnapshot | None:
    snap = fetch_akshare_spot(main_code)
    if snap:
        print(f"[intraday] {main_code} <- {snap.source} last={snap.last}")
        return snap
    snap = fetch_akshare_main_sina(main_code)
    if snap:
        print(f"[intraday] {main_code} <- {snap.source} last={snap.last}")
    return snap


def fetch_cu_snapshot() -> IntradaySnapshot | None:
    return fetch_snapshot("CU0")


def fetch_sc_snapshot() -> IntradaySnapshot | None:
    return fetch_snapshot("SC0")


def intraday_enabled() -> bool:
    """是否拉取盘中快照并在推送中展示。"""
    return load_env("INTRADAY_OVERLAY", "1").strip().lower() in ("1", "true", "yes")


def intraday_adjust_prob_enabled() -> bool:
    """是否用盘中数据修正 1 日概率（回测未提升准确率，默认关闭）。"""
    return load_env("INTRADAY_ADJUST_PROB", "0").strip().lower() in ("1", "true", "yes")


@dataclass
class IntradayBundle:
    cu: IntradaySnapshot | None = None
    oil: IntradaySnapshot | None = None
    wti: IntradaySnapshot | None = None


def fetch_all_snapshots() -> IntradayBundle:
    if not intraday_enabled():
        print("[intraday] INTRADAY_OVERLAY=0，跳过盘中快照")
        return IntradayBundle()
    return IntradayBundle(
        cu=fetch_cu_snapshot(),
        oil=fetch_sc_snapshot(),
        wti=fetch_wti_realtime(),
    )
