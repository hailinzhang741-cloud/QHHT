"""
沪铜 + 原油 联合每日预测 — 四个时段推送。

用法:
  python daily_multi_forecast.py
  python daily_multi_forecast.py --refresh

环境变量 RUN_SLOT: 0840 / 1030 / 1415 / 2050（计划任务自动设置）
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

from config import DATA_DIR, load_env
from engine import fetch_macro_data, get_engine
from engine_oil import get_oil_engine
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from forecast_oil_v5 import load_oil_data
from fundamentals import aggregate_holding, aggregate_wsr
from lme_ratio import fetch_lme_ratio_series
from notify_multi import notify_multi_if_signal
from notify_state import resolve_run_slot


LOG_DIR = DATA_DIR / "logs"
CU_HISTORY = DATA_DIR / "results" / "forecast_history.csv"
OIL_HISTORY = DATA_DIR / "oil" / "results" / "forecast_history.csv"


def parse_args():
    p = argparse.ArgumentParser(description="沪铜+原油联合每日预测")
    p.add_argument("--years", type=int, default=5, help="历史年数")
    p.add_argument("--no-refresh", action="store_true", help="使用本地缓存（默认每次重新拉取最新数据）")
    p.add_argument("--quiet", action="store_true", help="少输出")
    p.add_argument("--force-notify", action="store_true", help="强制推送（忽略信号过滤）")
    return p.parse_args()


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    log_file = LOG_DIR / "daily_multi_forecast.log"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def _append_history(result, path, run_time: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_time": run_time,
        "slot": resolve_run_slot(),
        "as_of_date": result.as_of_date,
        "close": result.close,
        "prob_up_1d": result.prob_up_1d,
        "prob_down_1d": result.prob_down_1d,
        "prob_up_5d": result.prob_up_5d,
        "prob_down_5d": result.prob_down_5d,
        "signal_1d": result.signal_1d,
        "signal_5d": result.signal_5d,
    }
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        slot_now = resolve_run_slot()
        if "slot" not in df_old.columns:
            df_old["slot"] = ""
        df_old = df_old[
            ~((df_old["as_of_date"].astype(str) == str(result.as_of_date)) & (df_old["slot"].astype(str) == slot_now))
        ]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slot = resolve_run_slot()
    use_cache = args.no_refresh
    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")

    try:
        cu_engine, cu_cfg, cu_ver = get_engine()
        oil_engine, oil_cfg, oil_ver = get_oil_engine()
        _log(f"开始联合预测 slot={slot} refresh={'否' if use_cache else '是'} (铜v{cu_ver} + 油v{oil_ver})")

        df_cu = fetch_cu_daily(start_date=start, use_cache=use_cache)
        wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
        hold = fetch_cu_holding_series(start, use_cache=use_cache)
        wsr_d = aggregate_wsr(wsr)
        hold_d = aggregate_holding(hold)
        lme_d = fetch_lme_ratio_series(df_cu, start, use_cache=use_cache)
        inv, pmi = fetch_macro_data(start, use_cache, cu_ver)

        if cu_ver >= 5:
            cu_result = cu_engine.forecast_cu(df_cu, wsr_d, hold_d, lme_d, inv, pmi, cfg=cu_cfg)
        else:
            cu_result = cu_engine.forecast_cu(df_cu, wsr_d, hold_d, lme_d, cfg=cu_cfg)

        df_oil, wsr_o, hold_o, ratio, api, pmi_o = load_oil_data(start, use_cache=use_cache)
        oil_result = oil_engine.forecast_sc(df_oil, wsr_o, hold_o, ratio, api, pmi_o, cfg=oil_cfg)

        _append_history(cu_result, CU_HISTORY, run_time)
        _append_history(oil_result, OIL_HISTORY, run_time)

        run_source = load_env("RUN_SOURCE", "local").strip() or "local"
        pushed = notify_multi_if_signal(
            cu_result,
            oil_result,
            force=args.force_notify,
            source=run_source,
            check_dedupe=True,
            slot=slot,
        )
        if pushed:
            _log(f"已推送微信 slot={slot}")
        elif load_env("DAILY_NOTIFY", "").lower() in ("1", "true", "yes"):
            _log("DAILY_NOTIFY=1 但推送失败，请检查 PUSHPLUS_TOKEN")

        if not args.quiet:
            print(cu_engine.format_report(cu_result))
            print(oil_engine.format_report(oil_result))

        _log(
            f"完成 slot={slot} refresh={'否' if use_cache else '是'} "
            f"铜 as_of={cu_result.as_of_date} close={cu_result.close} "
            f"1d={cu_result.prob_up_1d*100:.1f}% 5d={cu_result.prob_up_5d*100:.1f}% | "
            f"油 as_of={oil_result.as_of_date} close={oil_result.close} "
            f"1d={oil_result.prob_up_1d*100:.1f}% 5d={oil_result.prob_up_5d*100:.1f}%"
        )
        return 0
    except Exception as exc:
        _log(f"失败 slot={slot}: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
