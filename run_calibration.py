"""
沪铜预测校准 + 阈值自动优化 v4

用法:
  python run_calibration.py
  python run_calibration.py --years 5 --step 3
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from calibration import (
    calibration_metrics,
    format_calibration_report,
    optimize_and_save_thresholds,
    run_calibration_backtest,
)
from config import DATA_DIR
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from forecast import prepare_features
from fundamentals import aggregate_holding, aggregate_wsr
from lme_ratio import fetch_lme_ratio_series
from model_config import save_config


def parse_args():
    p = argparse.ArgumentParser(description="沪铜预测校准 v4")
    p.add_argument("--years", type=int, default=5, help="回测年数")
    p.add_argument("--step", type=int, default=3, help="walk-forward 步长")
    p.add_argument("--min-train", type=int, default=252, help="最少训练样本")
    p.add_argument("--refresh", action="store_true", help="忽略缓存")
    return p.parse_args()


def main():
    args = parse_args()
    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")
    use_cache = not args.refresh

    df = fetch_cu_daily(start_date=start, use_cache=use_cache)
    wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
    hold = fetch_cu_holding_series(start, use_cache=use_cache)
    lme = fetch_lme_ratio_series(df, start, use_cache=use_cache)
    feat = prepare_features(df, aggregate_wsr(wsr), aggregate_holding(hold), lme)

    print(f"\n[calibrate v4] years={args.years}, step={args.step}, long_only=True ...")
    cal1 = run_calibration_backtest(feat, min_train=args.min_train, step=args.step, horizon=1)
    cal5 = run_calibration_backtest(feat, min_train=args.min_train, step=args.step, horizon=5)

    cfg = optimize_and_save_thresholds(cal1, cal5, long_only=True)
    save_config(cfg)
    print(f"\n[optimize] 阈值 -> {DATA_DIR / 'results' / 'model_config.json'}")
    print(f"  5日做多阈值: ≥{cfg['filter']['5d']['th_up']*100:.0f}%, 子模型一致≥{cfg['filter']['5d']['min_agree']}")

    m1 = calibration_metrics(cal1, cfg["filter"]["1d"], long_only=True)
    m5 = calibration_metrics(cal5, cfg["filter"]["5d"], long_only=True)
    print(format_calibration_report(m1, m5))

    out_dir = DATA_DIR / "results"
    cal1.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_1d.csv", index=False, encoding="utf-8-sig")
    cal5.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_5d.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] 校准结果 -> {out_dir}")


if __name__ == "__main__":
    main()
