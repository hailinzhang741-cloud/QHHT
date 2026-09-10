"""
沪铜期货：Tushare 数据拉取 + 双均线回测

用法:
  1. 设置 token（任选其一）:
     - 环境变量: set TUSHARE_TOKEN=你的token
     - 或创建 .env 文件: TUSHARE_TOKEN=你的token

  2. 安装依赖:
     pip install -r requirements.txt

  3. 运行:
     python run_backtest.py
     python run_backtest.py --start 20200101 --fast 10 --slow 30
     python run_backtest.py --mode stitched   # 主力拼接（更准确）
     python run_backtest.py --holding         # 额外拉持仓排名
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backtest import BacktestConfig, plot_equity, print_report, run_backtest
from config import DATA_DIR
from fetcher import fetch_cu_daily, fetch_cu_holding, fetch_cu_main_stitched


def parse_args():
    p = argparse.ArgumentParser(description="沪铜期货 Tushare 回测")
    p.add_argument("--start", default="20180101", help="开始日期 YYYYMMDD")
    p.add_argument("--end", default=None, help="结束日期 YYYYMMDD")
    p.add_argument(
        "--mode",
        choices=["continuous", "main", "stitched"],
        default="continuous",
        help="数据源: continuous=CUL主连, main=CU主力, stitched=映射拼接",
    )
    p.add_argument("--fast", type=int, default=5, help="快均线周期")
    p.add_argument("--slow", type=int, default=20, help="慢均线周期")
    p.add_argument("--capital", type=float, default=500_000, help="初始资金")
    p.add_argument("--lots", type=int, default=1, help="交易手数")
    p.add_argument("--no-short", action="store_true", help="禁止做空，仅做多")
    p.add_argument("--no-cache", action="store_true", help="忽略本地缓存，重新拉取")
    p.add_argument("--plot", action="store_true", help="保存权益曲线图")
    p.add_argument("--holding", action="store_true", help="拉取最新持仓排名")
    return p.parse_args()


def main():
    args = parse_args()
    use_cache = not args.no_cache

    print("=" * 50)
    print("  沪铜(CU) Tushare 数据拉取 + 回测")
    print("=" * 50)

    # --- 拉数据 ---
    if args.mode == "stitched":
        df = fetch_cu_main_stitched(args.start, args.end, use_cache=use_cache)
    else:
        mode = "continuous" if args.mode == "continuous" else "main"
        df = fetch_cu_daily(args.start, args.end, mode=mode, use_cache=use_cache)

    print(f"\n数据范围: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}，共 {len(df)} 条")
    print(f"最新收盘: {df['close'].iloc[-1]:.0f} 元/吨，持仓量: {df['oi'].iloc[-1]:.0f} 手")

    if args.holding:
        hold = fetch_cu_holding()
        if not hold.empty:
            print("\n--- 最新会员持仓排名 (Top 10) ---")
            print(hold.head(10).to_string(index=False))

    # --- 回测 ---
    cfg = BacktestConfig(
        fast_period=args.fast,
        slow_period=args.slow,
        initial_capital=args.capital,
        lots=args.lots,
        allow_short=not args.no_short,
    )

    if cfg.fast_period >= cfg.slow_period:
        raise ValueError("快均线周期必须小于慢均线周期")

    result = run_backtest(df, cfg)
    print_report(result)

    # 保存结果
    out_dir = DATA_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"ma{args.fast}_{args.slow}_{args.mode}"
    result.trades.to_csv(out_dir / f"trades_{tag}.csv", index=False, encoding="utf-8-sig")
    result.equity_curve.to_csv(out_dir / f"equity_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] 交易记录 -> {out_dir / f'trades_{tag}.csv'}")
    print(f"[save] 权益曲线 -> {out_dir / f'equity_{tag}.csv'}")

    if args.plot:
        plot_path = out_dir / f"equity_{tag}.png"
        plot_equity(result, save_path=str(plot_path))


if __name__ == "__main__":
    main()
