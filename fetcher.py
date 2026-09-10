"""沪铜期货 Tushare 数据拉取模块。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd
import tushare as ts

from config import (
    CU_CONT_CODE,
    CU_MAIN_CODE,
    DATA_DIR,
    TUSHARE_TOKEN,
)

ContractMode = Literal["continuous", "main", "monthly"]


def get_pro():
    if not TUSHARE_TOKEN or TUSHARE_TOKEN == "your_tushare_token_here":
        raise ValueError(
            "未设置 TUSHARE_TOKEN。请在 .env 文件中配置，例如：\n"
            "  TUSHARE_TOKEN=your_token_here\n"
            "获取地址: https://tushare.pro/user/token"
        )
    # 显式传入 token，避免 ts.set_token 与系统环境变量冲突
    return ts.pro_api(TUSHARE_TOKEN)


def _paginate_fut_daily(pro, **kwargs) -> pd.DataFrame:
    """fut_daily 单次最多 2000 条，按日期分段拉取。"""
    start = kwargs.get("start_date")
    end = kwargs.get("end_date")
    if not start or not end:
        df = pro.fut_daily(**kwargs)
        return df if df is not None else pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=3), end_ts)
        part = pro.fut_daily(
            **{
                **kwargs,
                "start_date": cur.strftime("%Y%m%d"),
                "end_date": chunk_end.strftime("%Y%m%d"),
            }
        )
        if part is not None and not part.empty:
            chunks.append(part)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.15)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=["ts_code", "trade_date"]
    )


def fetch_cu_daily(
    start_date: str = "20180101",
    end_date: str | None = None,
    mode: ContractMode = "continuous",
    ts_code: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    拉取沪铜日线行情。

    mode:
      - continuous: CUL.SHF 主连（回测推荐）
      - main:       CU.SHF  主力
      - monthly:    指定月合约，如 CU2601.SHF
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    if ts_code is None:
        ts_code = CU_CONT_CODE if mode == "continuous" else CU_MAIN_CODE

    cache_path = DATA_DIR / f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 读取 {cache_path.name}，共 {len(df)} 条")
        return _normalize_daily(df)

    print(f"[fetch] 拉取 {ts_code}  {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_fut_daily(
        pro,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        raise RuntimeError(f"未获取到数据: {ts_code}")

    df = _normalize_daily(df)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[save]  已保存 {cache_path.name}，共 {len(df)} 条")
    return df


def fetch_cu_main_stitched(
    start_date: str = "20180101",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    通过 fut_mapping 拼接主力月合约日线（换月更准确，但请求较多）。
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    cache_path = DATA_DIR / f"CU_stitched_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 读取 {cache_path.name}，共 {len(df)} 条")
        return _normalize_daily(df)

    pro = get_pro()
    print(f"[fetch] 拉取主力映射 CU.SHF  {start_date} ~ {end_date} ...")

    mapping_chunks: list[pd.DataFrame] = []
    cur = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=2), end_ts)
        part = pro.fut_mapping(
            ts_code=CU_MAIN_CODE,
            start_date=cur.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
        )
        if part is not None and not part.empty:
            mapping_chunks.append(part)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.15)

    if not mapping_chunks:
        raise RuntimeError("未获取到 fut_mapping 数据")

    mapping = (
        pd.concat(mapping_chunks, ignore_index=True)
        .drop_duplicates(subset=["trade_date"])
        .sort_values("trade_date")
    )

    # 按映射合约分组批量拉行情
    rows: list[pd.DataFrame] = []
    for contract, grp in mapping.groupby("mapping_ts_code"):
        c_start = grp["trade_date"].min()
        c_end = grp["trade_date"].max()
        print(f"  -> {contract}  {c_start} ~ {c_end}")
        part = _paginate_fut_daily(
            pro,
            ts_code=contract,
            start_date=c_start,
            end_date=c_end,
        )
        if part is not None and not part.empty:
            rows.append(part)
        time.sleep(0.15)

    if not rows:
        raise RuntimeError("拼接失败：无合约行情")

    all_daily = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["trade_date", "ts_code"]
    )
    merged = mapping.merge(
        all_daily,
        left_on=["trade_date", "mapping_ts_code"],
        right_on=["trade_date", "ts_code"],
        how="inner",
    )
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged["ts_code"] = CU_MAIN_CODE

    df = _normalize_daily(merged)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[save]  已保存 {cache_path.name}，共 {len(df)} 条")
    return df


def fetch_cu_holding(trade_date: str | None = None, top_n: int = 20) -> pd.DataFrame:
    """拉取指定日期的会员持仓排名（多空力量参考）。"""
    if trade_date is None:
        trade_date = pd.Timestamp.today().strftime("%Y%m%d")
    pro = get_pro()
    df = pro.fut_holding(symbol="CU", trade_date=trade_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(top_n)


def fetch_cu_wsr(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """拉取仓单日报（库存压力参考）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    pro = get_pro()
    df = pro.fut_wsr(symbol="CU", start_date=start_date, end_date=end_date)
    return df if df is not None else pd.DataFrame()


def _paginate_by_year(pro, api_func, start_date: str, end_date: str, sleep_s: float = 0.15, **kwargs) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cur = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=1), end_ts)
        part = api_func(start_date=cur.strftime("%Y%m%d"), end_date=chunk_end.strftime("%Y%m%d"), **kwargs)
        if part is not None and not part.empty:
            chunks.append(part)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(sleep_s)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def fetch_cu_wsr_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """拉取仓单历史序列（按年分页，带缓存）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache_path = DATA_DIR / f"CU_wsr_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 仓单 {cache_path.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] 仓单 CU  {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_by_year(
        pro,
        lambda **kw: pro.fut_wsr(symbol="CU", **kw),
        start_date,
        end_date,
    )
    if not df.empty:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[save]  仓单 -> {cache_path.name}，共 {len(df)} 条")
    return df


def fetch_cu_holding_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """拉取会员持仓历史序列（按年分页，带缓存）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache_path = DATA_DIR / f"CU_holding_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"trade_date": str})
        print(f"[cache] 持仓 {cache_path.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] 持仓 CU  {start_date} ~ {end_date} ...")
    pro = get_pro()
    df = _paginate_by_year(
        pro,
        lambda **kw: pro.fut_holding(symbol="CU", exchange="SHFE", **kw),
        start_date,
        end_date,
    )
    if not df.empty:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[save]  持仓 -> {cache_path.name}，共 {len(df)} 条")
    return df


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "settle", "vol", "oi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


if __name__ == "__main__":
    df = fetch_cu_daily(start_date="20200101")
    print(df.tail())
    print(f"\n字段: {list(df.columns)}")
