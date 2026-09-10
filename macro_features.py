"""v5 宏观特征：LME 前日涨跌、LME 库存、中国 PMI。"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

from config import DATA_DIR
from fetcher import get_pro
from lookahead import prev_month

# akshare macro_euro_lme_stock 列序固定，铜库存为第 17 列（0-based index 16）
_LME_CU_STOCK_COL_IDX = 16
_PMI_MFG_COL = "PMI010000"


def _copper_stock_column(cols: list[str]) -> str:
    for i, c in enumerate(cols):
        if i == _LME_CU_STOCK_COL_IDX:
            return c
    raise ValueError("未找到 LME 铜库存列")


def fetch_lme_inventory_series(
    start_date: str,
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """LME 铜注册库存（akshare macro_euro_lme_stock）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache = DATA_DIR / f"LME_CU_stock_{start_date}_{end_date}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"trade_date": str})
        print(f"[cache] LME库存 {cache.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] LME 铜库存 {start_date} ~ {end_date} ...")
    raw = ak.macro_euro_lme_stock()
    cu_col = _copper_stock_column(raw.columns.tolist())
    df = raw[["日期", cu_col]].copy()
    df = df.rename(columns={"日期": "trade_date", cu_col: "lme_cu_stock"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    df["lme_cu_stock"] = pd.to_numeric(df["lme_cu_stock"], errors="coerce")
    df = df.dropna(subset=["trade_date", "lme_cu_stock"])
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[save]  LME库存 -> {cache.name}，共 {len(df)} 条")
    return df


def fetch_cn_pmi_series(
    start_date: str,
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """中国制造业 PMI（Tushare cn_pmi，月度）。"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    start_m = start_date[:6]
    end_m = end_date[:6]
    cache = DATA_DIR / f"CN_PMI_{start_m}_{end_m}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"month": str})
        print(f"[cache] PMI {cache.name}，共 {len(df)} 条")
        return df

    print(f"[fetch] 中国 PMI {start_m} ~ {end_m} ...")
    pro = get_pro()
    raw = pro.cn_pmi(start_m=start_m, end_m=end_m)
    time.sleep(0.12)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["month", "pmi_mfg", "pmi_chg1"])

    df = raw[["MONTH", _PMI_MFG_COL]].copy()
    df = df.rename(columns={"MONTH": "month", _PMI_MFG_COL: "pmi_mfg"})
    df["month"] = df["month"].astype(str)
    df["pmi_mfg"] = pd.to_numeric(df["pmi_mfg"], errors="coerce")
    df = df.dropna(subset=["month", "pmi_mfg"]).sort_values("month").reset_index(drop=True)
    df["pmi_chg1"] = df["pmi_mfg"].diff()
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[save]  PMI -> {cache.name}，共 {len(df)} 条")
    return df


def enrich_lme_return(feat: pd.DataFrame) -> pd.DataFrame:
    """从已合并的 LME 美元价计算前日/5日涨跌。"""
    out = feat.copy()
    if "lme_close_usd" not in out.columns:
        return out
    out["lme_ret1"] = out["lme_close_usd"].pct_change(1)
    out["lme_ret5"] = out["lme_close_usd"].pct_change(5)
    return out


def _enrich_inventory_features(inv: pd.DataFrame) -> pd.DataFrame:
    out = inv.copy()
    out["lme_inv_chg5"] = out["lme_cu_stock"].pct_change(5)
    ma60 = out["lme_cu_stock"].rolling(60, min_periods=20).mean()
    std60 = out["lme_cu_stock"].rolling(60, min_periods=20).std().replace(0, pd.NA)
    out["lme_inv_z60"] = (out["lme_cu_stock"] - ma60) / std60
    return out


def merge_macro_features(
    feat: pd.DataFrame,
    lme_inventory: pd.DataFrame | None = None,
    pmi_monthly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = enrich_lme_return(feat)

    if lme_inventory is not None and not lme_inventory.empty:
        inv = _enrich_inventory_features(lme_inventory)
        cols = ["trade_date", "lme_cu_stock", "lme_inv_chg5", "lme_inv_z60"]
        out = out.merge(inv[cols], on="trade_date", how="left")
        out["lme_cu_stock"] = out["lme_cu_stock"].ffill()
        out["lme_inv_chg5"] = out["lme_inv_chg5"].ffill()
        out["lme_inv_z60"] = out["lme_inv_z60"].ffill()

    if pmi_monthly is not None and not pmi_monthly.empty:
        pmi = pmi_monthly.copy()
        # M 月 PMI 约 M+1 月初发布；T 日所在月仅可用上月的 PMI
        out["pmi_month"] = out["trade_date"].astype(str).str[:6].map(prev_month)
        out = out.merge(
            pmi[["month", "pmi_mfg", "pmi_chg1"]].rename(columns={"month": "pmi_month"}),
            on="pmi_month",
            how="left",
        )
        out["pmi_mfg"] = out["pmi_mfg"].ffill()
        out["pmi_chg1"] = out["pmi_chg1"].ffill()
        out = out.drop(columns=["pmi_month"], errors="ignore")

    return out


def macro_prob(row: pd.Series) -> float:
    """LME 涨跌 + 库存 + PMI -> 偏多概率。"""
    votes: list[float] = []

    r1 = row.get("lme_ret1")
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

    inv5 = row.get("lme_inv_chg5")
    if pd.notna(inv5):
        if inv5 < -0.04:
            votes.append(0.58)  # 库存去化
        elif inv5 < -0.01:
            votes.append(0.54)
        elif inv5 > 0.04:
            votes.append(0.42)  # 累库
        elif inv5 > 0.01:
            votes.append(0.46)
        else:
            votes.append(0.50)

    pmi = row.get("pmi_mfg")
    if pd.notna(pmi):
        if pmi >= 51.0:
            votes.append(0.57)
        elif pmi >= 50.2:
            votes.append(0.54)
        elif pmi <= 48.5:
            votes.append(0.43)
        elif pmi <= 49.5:
            votes.append(0.46)
        else:
            votes.append(0.50)

    pmi_d = row.get("pmi_chg1")
    if pd.notna(pmi_d):
        if pmi_d >= 0.8:
            votes.append(0.55)
        elif pmi_d <= -0.8:
            votes.append(0.45)

    return float(sum(votes) / len(votes)) if votes else 0.5


def macro_summary(row: pd.Series) -> dict:
    parts: dict = {}
    if pd.notna(row.get("lme_ret1")):
        r = float(row["lme_ret1"]) * 100
        parts["LME前日涨跌_%"] = round(r, 2)
        parts["LME涨跌信号"] = "偏多" if r > 0.3 else ("偏空" if r < -0.3 else "中性")
    if pd.notna(row.get("lme_cu_stock")):
        parts["LME铜库存_吨"] = int(row["lme_cu_stock"])
    if pd.notna(row.get("lme_inv_chg5")):
        chg = float(row["lme_inv_chg5"]) * 100
        parts["LME库存5日变化_%"] = round(chg, 2)
        parts["LME库存信号"] = "偏多(去化)" if chg < -1 else ("偏空(累库)" if chg > 1 else "中性")
    if pd.notna(row.get("pmi_mfg")):
        parts["制造业PMI"] = round(float(row["pmi_mfg"]), 1)
        pmi = float(row["pmi_mfg"])
        parts["PMI信号"] = "扩张" if pmi >= 50 else "收缩"
    parts["宏观综合_上涨率"] = round(macro_prob(row) * 100, 1)
    return parts
