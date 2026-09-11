"""
原油(SC) 预测校准 + 阈值自动优化 v5

用法:
  python run_calibration_oil_v5.py
  python run_calibration_oil_v5.py --years 5 --step 3 --refresh
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from calibration import calibration_metrics
from calibration_oil_v5 import (
    eval_filter_v5,
    format_calibration_report,
    optimize_and_save_thresholds,
    run_calibration_backtest,
)
from config_oil import OIL_DATA_DIR
from forecast_oil_v5 import load_oil_data, prepare_features
from model_config_oil import save_config


def parse_args():
    p = argparse.ArgumentParser(description="原油(SC) 预测校准 v5")
    p.add_argument("--years", type=int, default=5, help="回测年数")
    p.add_argument("--step", type=int, default=3, help="walk-forward 步长")
    p.add_argument("--min-train", type=int, default=200, help="最少训练样本")
    p.add_argument("--refresh", action="store_true", help="忽略缓存")
    return p.parse_args()


def _metrics_v5(cal5: pd.DataFrame, cfg: dict, long_only: bool = True) -> dict:
    filter_cfg = cfg["filter"]["5d"]
    m = calibration_metrics(cal5, None, long_only=long_only)
    fd = eval_filter_v5(
        cal5,
        filter_cfg.get("th_up", 0.57),
        filter_cfg.get("min_agree", 4),
        cfg.get("quality_filter"),
        long_only=long_only,
    )
    m.update(
        {
            "filtered_accuracy": fd["filtered_accuracy"],
            "filtered_n_signals": fd["n_signals"],
            "filtered_n_long": fd["n_long"],
            "filtered_long_hit": fd["long_hit_rate"],
        }
    )
    return m


def main():
    args = parse_args()
    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")
    use_cache = not args.refresh

    df, wsr_d, hold_d, ratio, api, pmi = load_oil_data(start, use_cache=use_cache)
    feat = prepare_features(df, wsr_d, hold_d, ratio, api, pmi)

    print(f"\n[calibrate oil v5] years={args.years}, step={args.step}, long_only=True ...")
    cal1 = run_calibration_backtest(feat, min_train=args.min_train, step=args.step, horizon=1)
    cal5 = run_calibration_backtest(feat, min_train=args.min_train, step=args.step, horizon=5)

    cfg = optimize_and_save_thresholds(cal1, cal5, long_only=True)
    save_config(cfg)
    print(f"\n[optimize] 阈值 -> {OIL_DATA_DIR / 'results' / 'model_config_v5.json'}")
    print(f"  5日做多阈值: ≥{cfg['filter']['5d']['th_up']*100:.0f}%, 子模型一致≥{cfg['filter']['5d']['min_agree']}")

    m1 = calibration_metrics(cal1, cfg["filter"]["1d"], long_only=True)
    m5 = _metrics_v5(cal5, cfg, long_only=True)
    print(format_calibration_report(m1, m5))
    if cfg.get("quality_filter"):
        q = cfg["quality_filter"]
        print(f"  质量过滤: RSI≤{q.get('max_rsi14')}, 20日位置≤{q.get('max_price_pos20', 0)*100:.0f}%")

    out_dir = OIL_DATA_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cal1.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_v5_1d.csv", index=False, encoding="utf-8-sig")
    cal5.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_v5_5d.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] 校准结果 -> {out_dir}")


if __name__ == "__main__":
    main()
