"""原油外部特征：WTI、美国 API 库存、PMI。"""
from __future__ import annotations

import time

import akshare as ak
import pandas as pd

from config_oil import OIL_DATA_DIR
from lme_ratio import fetch_usdcnh_series
from lookahead import prev_month

WTI_SYMBOL = "CL"


def fetch_wti_usd_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = OIL_DATA_DIR / f"WTI_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] WTI {cache.name}，共 {len(df)} 条")
        return df
    print(f"[fetch] WTI {start_date} ~ {end_date} ...")
    raw = ak.futures_foreign_hist(symbol=WTI_SYMBOL)
    raw = raw.rename(columns={"date": "trade_date", "close": "wti_close_usd"})
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.strftime("%Y%m%d")
    df = raw[(raw["trade_date"] >= start_date) & (raw["trade_date"] <= end_date)]
    df = df[["trade_date", "wti_close_usd"]].dropna().sort_values("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def fetch_us_api_crude_series(start_date: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = OIL_DATA_DIR / f"US_API_crude_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] US API库存 {cache.name}，共 {len(df)} 条")
        return df
    print(f"[fetch] 美国 API 原油库存 ...")
    raw = ak.macro_usa_api_crude_stock()
    if "trade_date" not in raw.columns:
        cols = list(raw.columns)
        rename_map = {}
        if len(cols) >= 2:
            rename_map[cols[1]] = "trade_date"
        if len(cols) >= 3:
            rename_map[cols[2]] = "us_api_crude"
        raw = raw.rename(columns=rename_map)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    raw["us_api_crude"] = pd.to_numeric(raw["us_api_crude"], errors="coerce")
    df = raw.dropna(subset=["trade_date", "us_api_crude"])
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
    df["us_api_chg5"] = df["us_api_crude"].pct_change(5)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def build_sc_wti_ratio(sc_daily: pd.DataFrame, wti: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    base = sc_daily[["trade_date", "close"]].copy().rename(columns={"close": "sc_close"})
    base["trade_date"] = base["trade_date"].astype(str)
    merged = base.merge(wti, on="trade_date", how="left").merge(fx, on="trade_date", how="left")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged["wti_close_usd"] = merged["wti_close_usd"].ffill()
    merged["usdcnh"] = merged["usdcnh"].ffill()
    merged["wti_cny"] = merged["wti_close_usd"] * merged["usdcnh"]
    merged["sc_wti_ratio"] = merged["sc_close"] / merged["wti_cny"]
    merged["ratio_ma60"] = merged["sc_wti_ratio"].rolling(60, min_periods=20).mean()
    merged["ratio_std60"] = merged["sc_wti_ratio"].rolling(60, min_periods=20).std()
    merged["ratio_z60"] = (merged["sc_wti_ratio"] - merged["ratio_ma60"]) / merged["ratio_std60"].replace(0, pd.NA)
    merged["ratio_chg5"] = merged["sc_wti_ratio"].pct_change(5)
    merged["wti_ret1"] = merged["wti_close_usd"].pct_change(1)
    merged["wti_ret5"] = merged["wti_close_usd"].pct_change(5)
    return merged


def fetch_sc_wti_ratio_series(
    sc_daily: pd.DataFrame, start_date: str, end_date: str | None = None, use_cache: bool = True
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = OIL_DATA_DIR / f"sc_wti_ratio_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, dtype={"trade_date": str})
    wti = fetch_wti_usd_series(start_date, end_date, use_cache)
    fx = fetch_usdcnh_series(start_date, end_date, use_cache)
    df = build_sc_wti_ratio(sc_daily, wti, fx)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def merge_oil_external(feat: pd.DataFrame, ratio: pd.DataFrame, api: pd.DataFrame, pmi_monthly: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    if ratio is not None and not ratio.empty:
        cols = [c for c in ratio.columns if c != "trade_date"]
        out = out.merge(ratio[["trade_date"] + cols], on="trade_date", how="left")
        for c in cols:
            if c in out.columns:
                out[c] = out[c].ffill()
    if api is not None and not api.empty:
        out = out.merge(api[["trade_date", "us_api_crude", "us_api_chg5"]], on="trade_date", how="left")
        out["us_api_crude"] = out["us_api_crude"].ffill()
        out["us_api_chg5"] = out["us_api_chg5"].ffill()
    if pmi_monthly is not None and not pmi_monthly.empty:
        out["pmi_month"] = out["trade_date"].astype(str).str[:6].map(prev_month)
        out = out.merge(
            pmi_monthly[["month", "pmi_mfg", "pmi_chg1"]].rename(columns={"month": "pmi_month"}),
            on="pmi_month",
            how="left",
        )
        out["pmi_mfg"] = out["pmi_mfg"].ffill()
        out["pmi_chg1"] = out["pmi_chg1"].ffill()
        out = out.drop(columns=["pmi_month"], errors="ignore")
    return out


def oil_macro_prob(row: pd.Series) -> float:
    votes: list[float] = []
    r1 = row.get("wti_ret1")
    if pd.notna(r1):
        if r1 > 0.015:
            votes.append(0.60)
        elif r1 > 0.003:
            votes.append(0.56)
        elif r1 < -0.015:
            votes.append(0.40)
        elif r1 < -0.003:
            votes.append(0.44)
        else:
            votes.append(0.50)
    z = row.get("ratio_z60")
    if pd.notna(z):
        if z < -0.8:
            votes.append(0.58)
        elif z < -0.2:
            votes.append(0.54)
        elif z > 0.8:
            votes.append(0.42)
        elif z > 0.2:
            votes.append(0.46)
        else:
            votes.append(0.50)
    chg = row.get("us_api_chg5")
    if pd.notna(chg):
        if chg < -0.03:
            votes.append(0.57)
        elif chg < -0.01:
            votes.append(0.53)
        elif chg > 0.03:
            votes.append(0.43)
        elif chg > 0.01:
            votes.append(0.47)
    pmi = row.get("pmi_mfg")
    if pd.notna(pmi):
        if pmi >= 51:
            votes.append(0.55)
        elif pmi <= 48.5:
            votes.append(0.44)
    return float(sum(votes) / len(votes)) if votes else 0.5


def oil_ratio_prob(row: pd.Series) -> float:
    votes: list[float] = []
    z = row.get("ratio_z60")
    if pd.notna(z):
        if z < -0.8:
            votes.append(0.59)
        elif z < -0.2:
            votes.append(0.55)
        elif z > 0.8:
            votes.append(0.41)
        elif z > 0.2:
            votes.append(0.45)
        else:
            votes.append(0.50)
    chg = row.get("ratio_chg5")
    if pd.notna(chg):
        if chg < -0.02:
            votes.append(0.54)
        elif chg > 0.02:
            votes.append(0.46)
    return float(sum(votes) / len(votes)) if votes else 0.5


def oil_macro_summary(row: pd.Series) -> dict:
    parts: dict = {}
    if pd.notna(row.get("wti_ret1")):
        parts["WTI前日_%"] = round(float(row["wti_ret1"]) * 100, 2)
    if pd.notna(row.get("sc_wti_ratio")):
        parts["SC/WTI比"] = round(float(row["sc_wti_ratio"]), 4)
    if pd.notna(row.get("us_api_chg5")):
        parts["美库存5日_%"] = round(float(row["us_api_chg5"]) * 100, 2)
    if pd.notna(row.get("pmi_mfg")):
        parts["制造业PMI"] = round(float(row["pmi_mfg"]), 1)
    parts["宏观_上涨率"] = round(oil_macro_prob(row) * 100, 1)
    return parts
