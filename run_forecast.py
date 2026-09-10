"""
沪铜涨跌概率预测（v4 默认 / v5 可选）

用法:
  python run_forecast.py
  python run_forecast.py --refresh
  python run_forecast.py --v5              # 使用 v5 模型
  python run_forecast.py --notify        # 有5日做多信号时推送
  python run_forecast.py --test-notify   # 测试推送配置
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from config import DATA_DIR
from engine import fetch_macro_data, get_engine
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from fundamentals import aggregate_holding, aggregate_wsr
from lme_ratio import fetch_lme_ratio_series
from notify import notify_if_signal, test_notify


def parse_args():
    p = argparse.ArgumentParser(description="沪铜涨跌概率预测")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--v5", action="store_true", help="使用 v5 模型（LME涨跌/库存+PMI）")
    p.add_argument("--v4", action="store_true", help="强制使用 v4 模型")
    p.add_argument("--notify", action="store_true", help="5日做多信号时推送")
    p.add_argument("--test-notify", action="store_true", help="测试推送渠道")
    p.add_argument("--calibrate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.test_notify:
        test_notify()
        return

    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")
    use_cache = not args.refresh

    version = 5 if args.v5 else (4 if args.v4 else None)
    engine, cfg, ver = get_engine(version)
    print(f"[engine] 使用 v{ver}")

    df = fetch_cu_daily(start_date=start, use_cache=use_cache)
    wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
    hold = fetch_cu_holding_series(start, use_cache=use_cache)
    lme = fetch_lme_ratio_series(df, start, use_cache=use_cache)
    wsr_d = aggregate_wsr(wsr)
    hold_d = aggregate_holding(hold)
    inv, pmi = fetch_macro_data(start, use_cache, ver)

    if ver >= 5:
        result = engine.forecast_cu(df, wsr_d, hold_d, lme, inv, pmi, cfg=cfg)
    else:
        result = engine.forecast_cu(df, wsr_d, hold_d, lme, cfg=cfg)
    print(engine.format_report(result))

    out_dir = DATA_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([{
        "as_of_date": result.as_of_date,
        "close": result.close,
        "prob_up_5d": result.prob_up_5d,
        "signal_5d": result.signal_5d,
        "shfe_lme_ratio": result.signals.get("沪伦比"),
    }]).to_csv(out_dir / "latest_forecast.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] -> {out_dir / 'latest_forecast.csv'}")

    if args.notify or result.signal_5d == "做多":
        notify_if_signal(result)

    if args.calibrate:
        if ver >= 5:
            from run_calibration_v5 import main as cal_main
        else:
            from run_calibration import main as cal_main
        cal_main()


if __name__ == "__main__":
    main()
