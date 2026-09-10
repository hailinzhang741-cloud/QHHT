"""沪铜期货简单回测引擎 — 双均线策略示例。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import CU_MULTIPLIER, CU_TICK_VALUE


@dataclass
class BacktestConfig:
    fast_period: int = 5
    slow_period: int = 20
    initial_capital: float = 500_000.0   # 初始资金（元）
    lots: int = 1                          # 每次交易手数
    margin_rate: float = 0.11              # 保证金比例
    commission_rate: float = 0.000051      # 单边费率（约万分之0.51）
    commission_per_lot: float = 0.0        # 固定手续费/手（>0 时覆盖费率）
    slippage_ticks: int = 1                # 滑点（跳数）
    allow_short: bool = True               # 是否允许做空
    price_col: str = "close"               # 信号用收盘价
    exec_col: str = "open"                 # 次日开盘执行


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame


def ma_crossover_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """双均线金叉做多、死叉做空。"""
    out = df.copy()
    price = out[cfg.price_col]
    out["ma_fast"] = price.rolling(cfg.fast_period, min_periods=cfg.fast_period).mean()
    out["ma_slow"] = price.rolling(cfg.slow_period, min_periods=cfg.slow_period).mean()

    if cfg.allow_short:
        raw = np.where(out["ma_fast"] > out["ma_slow"], 1, -1)
    else:
        raw = np.where(out["ma_fast"] > out["ma_slow"], 1, 0)

    out["signal"] = pd.Series(raw, index=out.index, dtype=float)
    ma_ready = out["ma_fast"].notna() & out["ma_slow"].notna()
    out.loc[~ma_ready, "signal"] = np.nan
    out["signal"] = out["signal"].fillna(0).astype(int)
    return out


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    if cfg is None:
        cfg = BacktestConfig()

    if cfg.exec_col not in df.columns:
        raise ValueError(f"缺少执行价格列: {cfg.exec_col}")

    sig_df = ma_crossover_signals(df, cfg)
    sig_df["position"] = sig_df["signal"].shift(1).fillna(0).astype(int)

    exec_price = sig_df[cfg.exec_col]
    tick_slip = cfg.slippage_ticks * (CU_TICK_VALUE / CU_MULTIPLIER)  # 元/吨

    trades: list[dict] = []
    equity = cfg.initial_capital
    prev_pos = 0
    entry_price = 0.0
    entry_date = ""

    equity_rows: list[dict] = []

    for i, row in sig_df.iterrows():
        date = row["trade_date"]
        pos = int(row["position"])
        price = float(exec_price.iloc[i]) if pd.notna(exec_price.iloc[i]) else np.nan

        if np.isnan(price):
            equity_rows.append({"trade_date": date, "equity": equity, "position": prev_pos})
            continue

        # 换仓
        if pos != prev_pos:
            # 平旧仓
            if prev_pos != 0:
                exit_price = price - tick_slip * prev_pos  # 多头卖滑点向下，空头买滑点向上
                pnl = (exit_price - entry_price) * prev_pos * CU_MULTIPLIER * cfg.lots
                comm = _commission(cfg, entry_price) + _commission(cfg, exit_price)
                pnl -= comm
                equity += pnl
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": "LONG" if prev_pos > 0 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "lots": cfg.lots,
                        "pnl": pnl,
                        "commission": comm,
                    }
                )

            # 开新仓
            if pos != 0:
                entry_price = price + tick_slip * pos
                entry_date = date
                comm = _commission(cfg, entry_price)
                equity -= comm
            prev_pos = pos

        # 盯市浮盈（用 close 估算）
        mtm = 0.0
        if prev_pos != 0 and pd.notna(row.get("close")):
            mtm = (float(row["close"]) - entry_price) * prev_pos * CU_MULTIPLIER * cfg.lots

        margin = float(row["close"]) * CU_MULTIPLIER * cfg.lots * cfg.margin_rate if prev_pos else 0
        equity_rows.append(
            {
                "trade_date": date,
                "equity": equity + mtm,
                "cash": equity,
                "mtm": mtm,
                "position": prev_pos,
                "margin_used": margin,
                "close": row["close"],
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    metrics = _calc_metrics(equity_df, trades_df, cfg)
    return BacktestResult(metrics=metrics, equity_curve=equity_df, trades=trades_df, signals=sig_df)


def _commission(cfg: BacktestConfig, price: float) -> float:
    if cfg.commission_per_lot > 0:
        return cfg.commission_per_lot
    notional = price * CU_MULTIPLIER * cfg.lots
    return notional * cfg.commission_rate


def _calc_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if equity_df.empty:
        return {}

    eq = equity_df["equity"]
    ret = eq.pct_change().fillna(0)
    total_return = eq.iloc[-1] / cfg.initial_capital - 1

    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min()

    win_rate = 0.0
    avg_pnl = 0.0
    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / len(trades_df)
        avg_pnl = trades_df["pnl"].mean()

    sharpe = 0.0
    if ret.std() > 0:
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252)

    return {
        "initial_capital": cfg.initial_capital,
        "final_equity": round(float(eq.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "trade_count": len(trades_df),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_pnl_per_trade": round(float(avg_pnl), 2),
        "fast_ma": cfg.fast_period,
        "slow_ma": cfg.slow_period,
    }


def print_report(result: BacktestResult) -> None:
    print("\n" + "=" * 50)
    print("  沪铜双均线回测报告")
    print("=" * 50)
    for k, v in result.metrics.items():
        print(f"  {k:25s}: {v}")
    print("=" * 50)

    if not result.trades.empty:
        print("\n最近 5 笔交易:")
        print(result.trades.tail(5).to_string(index=False))


def plot_equity(result: BacktestResult, save_path: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    eq = result.equity_curve
    axes[0].plot(eq["trade_date"], eq["equity"], label="Equity", color="steelblue")
    axes[0].set_ylabel("权益 (元)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("沪铜 MA 策略权益曲线")

    sig = result.signals
    axes[1].plot(sig["trade_date"], sig["close"], label="Close", alpha=0.6)
    axes[1].plot(sig["trade_date"], sig["ma_fast"], label=f'MA{result.metrics.get("fast_ma")}', linewidth=1)
    axes[1].plot(sig["trade_date"], sig["ma_slow"], label=f'MA{result.metrics.get("slow_ma")}', linewidth=1)
    axes[1].set_ylabel("价格 (元/吨)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # x 轴稀疏显示
    for ax in axes:
        ticks = ax.get_xticks()
        labels = [eq["trade_date"].iloc[int(t)] if 0 <= int(t) < len(eq) else "" for t in ticks]
        ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"\n[plot] 图表已保存: {save_path}")
    else:
        plt.show()
