"""原油(SC) Tushare 数据拉取 — 独立于沪铜 fetcher。"""
from __future__ import annotations

import time

import pandas as pd

from config_oil import OIL_CONT_CODE, OIL_DATA_DIR, OIL_EXCHANGE, OIL_MAIN_CODE, OIL_SYMBOL
from fetcher import _normalize_daily, _paginate_by_year, _paginate_fut_daily, get_pro


def fetch_sc_daily(
    start_date: str = "20180301",
    end_date: str | None = None,
    use_cache: bool = True,
    continuous: bool = True,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    ts_code = OIL_CONT_CODE if continuous else OIL_MAIN_CODE
    cache_path = OIL_DATA_DIR / f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 原油 {cache_path.name}，共 {len(df)} 条")
        return _normalize_daily(df)

    print(f"[fetch] 原油 {ts_code} {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_fut_daily(pro, ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        raise RuntimeError(f"未获取到原油数据: {ts_code}")
    df = _normalize_daily(df)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[save]  原油 -> {cache_path.name}，共 {len(df)} 条")
    return df


def fetch_sc_wsr_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache_path = OIL_DATA_DIR / f"SC_wsr_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 原油仓单 {cache_path.name}，共 {len(df)} 条")
        return df
    print(f"[fetch] 原油仓单 {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_by_year(pro, lambda **kw: pro.fut_wsr(symbol=OIL_SYMBOL, **kw), start_date, end_date)
    if not df.empty:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def fetch_sc_holding_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache_path = OIL_DATA_DIR / f"SC_holding_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 原油持仓 {cache_path.name}，共 {len(df)} 条")
        return df
    print(f"[fetch] 原油持仓 {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_by_year(
        pro,
        lambda **kw: pro.fut_holding(symbol=OIL_SYMBOL, exchange=OIL_EXCHANGE, **kw),
        start_date,
        end_date,
    )
    if not df.empty:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df
