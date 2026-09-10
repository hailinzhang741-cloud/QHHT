"""沪铜基本面特征：仓单 + 会员持仓。"""
from __future__ import annotations

import pandas as pd


def aggregate_wsr(wsr: pd.DataFrame) -> pd.DataFrame:
    """仓单日报按日汇总。"""
    if wsr is None or wsr.empty:
        return pd.DataFrame(columns=["trade_date", "wsr_vol", "wsr_chg1", "wsr_chg5"])

    df = wsr.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    daily = df.groupby("trade_date", as_index=False)["vol"].sum()
    daily = daily.rename(columns={"vol": "wsr_vol"})
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["wsr_chg1"] = daily["wsr_vol"].diff()
    daily["wsr_chg5"] = daily["wsr_vol"].diff(5)
    daily["wsr_pct5"] = daily["wsr_vol"].pct_change(5)
    return daily


def aggregate_holding(holding: pd.DataFrame) -> pd.DataFrame:
    """持仓排名按日汇总（前 20 名会员合计）。"""
    if holding is None or holding.empty:
        return pd.DataFrame(
            columns=["trade_date", "long_total", "short_total", "net_long", "net_long_chg1", "net_long_chg5"]
        )

    df = holding.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    for col in ("long_hld", "short_hld", "long_chg", "short_chg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    daily = (
        df.groupby("trade_date", as_index=False)
        .agg(long_total=("long_hld", "sum"), short_total=("short_hld", "sum"), long_chg=("long_chg", "sum"), short_chg=("short_chg", "sum"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    # 全市场多头总量≈空头总量，净持仓水平恒为 0；用「增仓差」更有信号
    daily["net_position_chg"] = daily["long_chg"] - daily["short_chg"]
    daily["net_position_chg5"] = daily["net_position_chg"].rolling(5).sum()
    daily["long_chg5"] = daily["long_chg"].rolling(5).sum()
    daily["short_chg5"] = daily["short_chg"].rolling(5).sum()
    # 兼容旧字段名
    daily["net_long"] = daily["long_total"] - daily["short_total"]
    daily["net_long_chg5"] = daily["net_position_chg5"]
    return daily


def merge_fundamentals(price_feat: pd.DataFrame, wsr_daily: pd.DataFrame, hold_daily: pd.DataFrame) -> pd.DataFrame:
    out = price_feat.copy()
    if not wsr_daily.empty:
        out = out.merge(wsr_daily, on="trade_date", how="left")
    if not hold_daily.empty:
        out = out.merge(hold_daily, on="trade_date", how="left")
    return out


def fundamental_prob(row: pd.Series) -> float:
    """仓单 + 持仓 -> 偏多概率。"""
    votes: list[float] = []

    wsr5 = row.get("wsr_chg5")
    if pd.notna(wsr5):
        if wsr5 < -500:
            votes.append(0.58)
        elif wsr5 < 0:
            votes.append(0.54)
        elif wsr5 > 500:
            votes.append(0.42)
        elif wsr5 > 0:
            votes.append(0.46)
        else:
            votes.append(0.50)

    npc5 = row.get("net_position_chg5")
    if pd.notna(npc5):
        if npc5 > 500:
            votes.append(0.57)
        elif npc5 > 0:
            votes.append(0.54)
        elif npc5 < -500:
            votes.append(0.43)
        elif npc5 < 0:
            votes.append(0.46)
        else:
            votes.append(0.50)

    if pd.notna(row.get("long_chg5")) and pd.notna(row.get("short_chg5")):
        if row["long_chg5"] > 0 and row["short_chg5"] <= 0:
            votes.append(0.56)
        elif row["long_chg5"] < 0 and row["short_chg5"] > 0:
            votes.append(0.42)
        elif row["long_chg5"] < 0:
            votes.append(0.44)

    return float(sum(votes) / len(votes)) if votes else 0.5


def fundamental_summary(row: pd.Series) -> dict:
    parts = {}
    if pd.notna(row.get("wsr_vol")):
        parts["仓单总量_吨"] = int(row["wsr_vol"])
    if pd.notna(row.get("wsr_chg5")):
        chg = row["wsr_chg5"]
        parts["仓单5日变化_吨"] = int(chg)
        parts["仓单信号"] = "偏多(去化)" if chg < 0 else ("偏空(累库)" if chg > 0 else "中性")
    if pd.notna(row.get("net_position_chg5")):
        parts["净增仓5日差"] = int(row["net_position_chg5"])
        npc = row["net_position_chg5"]
        parts["持仓信号"] = "偏多(净增仓)" if npc > 0 else ("偏空(净减仓)" if npc < 0 else "中性")
    parts["基本面综合_上涨率"] = round(fundamental_prob(row) * 100, 1)
    return parts
