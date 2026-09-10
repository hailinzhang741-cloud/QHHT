"""沪伦铜比价 — LME(CAD) + USDCNH + 沪铜。"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

from config import DATA_DIR
from fetcher import get_pro

LME_SYMBOL = "CAD"  # akshare: LME 铜


def fetch_lme_usd_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """LME 铜美元价（akshare futures_foreign_hist）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = DATA_DIR / f"LME_CAD_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] LME {cache.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] LME 铜(CAD) {start_date} ~ {end_date} ...")
    raw = ak.futures_foreign_hist(symbol=LME_SYMBOL)
    raw = raw.rename(columns={"date": "trade_date"})
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.strftime("%Y%m%d")
    raw["lme_close_usd"] = pd.to_numeric(raw["close"], errors="coerce")
    df = raw[(raw["trade_date"] >= start_date) & (raw["trade_date"] <= end_date)]
    df = df[["trade_date", "lme_close_usd"]].dropna().sort_values("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[save]  LME -> {cache.name}，共 {len(df)} 条")
    return df


def fetch_usdcnh_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """离岸人民币 USDCNH（Tushare fx_daily）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = DATA_DIR / f"USDCNH_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] USDCNH {cache.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] USDCNH {start_date} ~ {end_date} ...")
    pro = get_pro()
    chunks: list[pd.DataFrame] = []
    cur = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while cur <= end_ts:
        ce = min(cur + pd.DateOffset(years=2), end_ts)
        part = pro.fx_daily(
            ts_code="USDCNH.FXCM",
            start_date=cur.strftime("%Y%m%d"),
            end_date=ce.strftime("%Y%m%d"),
        )
        if part is not None and not part.empty:
            chunks.append(part)
        cur = ce + pd.Timedelta(days=1)
        time.sleep(0.12)

    if not chunks:
        return pd.DataFrame(columns=["trade_date", "usdcnh"])
    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["trade_date"])
    df["trade_date"] = df["trade_date"].astype(str)
    df["usdcnh"] = (df["bid_close"] + df["ask_close"]) / 2
    df = df[["trade_date", "usdcnh"]].sort_values("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[save]  USDCNH -> {cache.name}，共 {len(df)} 条")
    return df


def build_shfe_lme_ratio(cu_daily: pd.DataFrame, lme: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    沪伦比 = 沪铜(元/吨) / (LME美元/吨 × USDCNH)
    比值偏高 → 国内偏贵(偏空)；偏低 → 国内偏便宜(偏多)
    """
    base = cu_daily[["trade_date", "close"]].copy()
    base = base.rename(columns={"close": "shfe_close"})
    base["trade_date"] = base["trade_date"].astype(str)

    merged = base.merge(lme, on="trade_date", how="left")
    merged = merged.merge(fx, on="trade_date", how="left")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged["lme_close_usd"] = merged["lme_close_usd"].ffill()
    merged["usdcnh"] = merged["usdcnh"].ffill()

    merged["lme_cny"] = merged["lme_close_usd"] * merged["usdcnh"]
    merged["shfe_lme_ratio"] = merged["shfe_close"] / merged["lme_cny"]
    merged["ratio_ma60"] = merged["shfe_lme_ratio"].rolling(60, min_periods=20).mean()
    merged["ratio_std60"] = merged["shfe_lme_ratio"].rolling(60, min_periods=20).std()
    merged["ratio_z60"] = (merged["shfe_lme_ratio"] - merged["ratio_ma60"]) / merged["ratio_std60"].replace(0, pd.NA)
    merged["ratio_chg5"] = merged["shfe_lme_ratio"].pct_change(5)
    return merged


def fetch_lme_ratio_series(
    cu_daily: pd.DataFrame,
    start_date: str,
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = DATA_DIR / f"shfe_lme_ratio_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] 沪伦比 {cache.name}，共 {len(df)} 条")
        return df

    lme = fetch_lme_usd_series(start_date, end_date, use_cache)
    fx = fetch_usdcnh_series(start_date, end_date, use_cache)
    df = build_shfe_lme_ratio(cu_daily, lme, fx)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[save]  沪伦比 -> {cache.name}，共 {len(df)} 条")
    return df


def merge_lme_ratio(feat: pd.DataFrame, ratio_daily: pd.DataFrame) -> pd.DataFrame:
    if ratio_daily is None or ratio_daily.empty:
        return feat
    cols = ["trade_date", "lme_close_usd", "usdcnh", "shfe_lme_ratio", "ratio_z60", "ratio_chg5"]
    cols = [c for c in cols if c in ratio_daily.columns]
    return feat.merge(ratio_daily[cols], on="trade_date", how="left")


def lme_ratio_prob(row: pd.Series) -> float:
    """沪伦比价 -> 偏多概率。"""
    votes: list[float] = []
    z = row.get("ratio_z60")
    if pd.notna(z):
        if z < -1.0:
            votes.append(0.60)  # 国内相对便宜
        elif z < -0.3:
            votes.append(0.56)
        elif z > 1.0:
            votes.append(0.40)  # 国内相对贵
        elif z > 0.3:
            votes.append(0.44)
        else:
            votes.append(0.50)

    chg = row.get("ratio_chg5")
    if pd.notna(chg):
        if chg < -0.02:
            votes.append(0.55)  # 比值回落，国内走强
        elif chg > 0.02:
            votes.append(0.45)
    return float(sum(votes) / len(votes)) if votes else 0.5


def lme_ratio_summary(row: pd.Series) -> dict:
    parts: dict = {}
    if pd.notna(row.get("shfe_lme_ratio")):
        parts["沪伦比"] = round(float(row["shfe_lme_ratio"]), 4)
    if pd.notna(row.get("ratio_z60")):
        z = float(row["ratio_z60"])
        parts["沪伦比Z60"] = round(z, 2)
        if z < -0.5:
            parts["沪伦比信号"] = "偏多(国内相对便宜)"
        elif z > 0.5:
            parts["沪伦比信号"] = "偏空(国内相对贵)"
        else:
            parts["沪伦比信号"] = "中性"
    if pd.notna(row.get("lme_close_usd")):
        parts["LME铜_美元"] = round(float(row["lme_close_usd"]), 1)
    parts["沪伦比_上涨率"] = round(lme_ratio_prob(row) * 100, 1)
    return parts
