"""
每日自动预测 — 拉最新数据、生成概率、追加历史记录。

用法:
  python daily_forecast.py
  python daily_forecast.py --refresh

Windows 计划任务: 见 scripts/register_daily_task.ps1（建议次日 08:30，数据更完整）
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import DATA_DIR, ROOT_DIR, load_env
from engine import fetch_macro_data, get_engine
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from fundamentals import aggregate_holding, aggregate_wsr
from lme_ratio import fetch_lme_ratio_series
from notify import notify_if_signal


LOG_DIR = DATA_DIR / "logs"
HISTORY_PATH = DATA_DIR / "results" / "forecast_history.csv"


def parse_args():
    p = argparse.ArgumentParser(description="沪铜每日自动预测")
    p.add_argument("--years", type=int, default=5, help="训练/统计使用的历史年数")
    p.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取")
    p.add_argument("--quiet", action="store_true", help="仅写日志，少输出")
    return p.parse_args()


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    log_file = LOG_DIR / "daily_forecast.log"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def _append_history(result, run_time: str) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_time": run_time,
        "as_of_date": result.as_of_date,
        "close": result.close,
        "prob_up_1d": result.prob_up_1d,
        "prob_down_1d": result.prob_down_1d,
        "prob_up_5d": result.prob_up_5d,
        "prob_down_5d": result.prob_down_5d,
        "signal_1d": result.signal_1d,
        "signal_5d": result.signal_5d,
        "fundamental_up_pct": result.signals.get("基本面_上涨率"),
        "wsr_signal": result.signals.get("仓单信号"),
        "holding_signal": result.signals.get("持仓信号"),
        "rsi14": result.signals.get("RSI14"),
    }
    df_new = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        df_old = pd.read_csv(HISTORY_PATH)
        # 同一交易日重复运行则覆盖
        df_old = df_old[df_old["as_of_date"].astype(str) != str(result.as_of_date)]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    use_cache = not args.refresh
    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")

    try:
        engine, cfg, ver = get_engine()
        _log(f"开始每日预测 (v{ver})")
        df = fetch_cu_daily(start_date=start, use_cache=use_cache)
        wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
        hold = fetch_cu_holding_series(start, use_cache=use_cache)
        wsr_d = aggregate_wsr(wsr)
        hold_d = aggregate_holding(hold)
        lme_d = fetch_lme_ratio_series(df, start, use_cache=use_cache)
        inv, pmi = fetch_macro_data(start, use_cache, ver)

        if ver >= 5:
            result = engine.forecast_cu(df, wsr_d, hold_d, lme_d, inv, pmi, cfg=cfg)
        else:
            result = engine.forecast_cu(df, wsr_d, hold_d, lme_d, cfg=cfg)
        _append_history(result, run_time)
        pushed = notify_if_signal(result)
        if pushed:
            _log("已推送微信")
        elif load_env("DAILY_NOTIFY", "").lower() in ("1", "true", "yes"):
            _log("DAILY_NOTIFY=1 但推送失败，请检查 PUSHPLUS_TOKEN")

        latest_path = DATA_DIR / "results" / "latest_forecast.csv"
        pd.DataFrame(
            [
                {
                    "run_time": run_time,
                    "as_of_date": result.as_of_date,
                    "close": result.close,
                    "prob_up_1d": result.prob_up_1d,
                    "prob_down_1d": result.prob_down_1d,
                    "prob_up_5d": result.prob_up_5d,
                    "prob_down_5d": result.prob_down_5d,
                }
            ]
        ).to_csv(latest_path, index=False, encoding="utf-8-sig")

        if not args.quiet:
            print(engine.format_report(result))
        _log(
            f"完成 as_of={result.as_of_date} "
            f"1d={result.prob_up_1d*100:.1f}% 5d={result.prob_up_5d*100:.1f}% "
            f"-> {HISTORY_PATH.name}"
        )
        return 0
    except Exception as exc:
        _log(f"失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
