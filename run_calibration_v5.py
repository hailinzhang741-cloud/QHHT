"""
沪铜预测校准 + 阈值自动优化 v5

用法:
  python run_calibration_v5.py
  python run_calibration_v5.py --years 5 --step 3
  python run_calibration_v5.py --compare   # 与 v4 对比
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from calibration import calibration_metrics, run_calibration_backtest as run_cal_v4
from calibration_v5 import (
    eval_filter_v5,
    format_calibration_report,
    optimize_and_save_thresholds,
    run_calibration_backtest,
)
from lookahead import audit_report_v4_v5
from config import DATA_DIR
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from forecast import prepare_features as prepare_features_v4
from forecast_v5 import prepare_features as prepare_features_v5
from fundamentals import aggregate_holding, aggregate_wsr
from lme_ratio import fetch_lme_ratio_series
from macro_features import fetch_cn_pmi_series, fetch_lme_inventory_series
from model_config import load_config, save_config
from signal_filter import eval_filter_detailed


def parse_args():
    p = argparse.ArgumentParser(description="沪铜预测校准 v5")
    p.add_argument("--years", type=int, default=5, help="回测年数")
    p.add_argument("--step", type=int, default=3, help="walk-forward 步长")
    p.add_argument("--min-train", type=int, default=252, help="最少训练样本")
    p.add_argument("--refresh", action="store_true", help="忽略缓存")
    p.add_argument("--compare", action="store_true", help="输出 v4 vs v5 对比")
    p.add_argument("--audit", action="store_true", help="输出未来函数审计报告")
    return p.parse_args()


def _metrics_v5(cal5: pd.DataFrame, cfg: dict, long_only: bool = True) -> dict:
    from calibration import calibration_metrics as base_metrics

    filter_cfg = cfg["filter"]["5d"]
    m = base_metrics(cal5, None, long_only=long_only)
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

    df = fetch_cu_daily(start_date=start, use_cache=use_cache)
    wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
    hold = fetch_cu_holding_series(start, use_cache=use_cache)
    lme = fetch_lme_ratio_series(df, start, use_cache=use_cache)
    wsr_d = aggregate_wsr(wsr)
    hold_d = aggregate_holding(hold)
    inv = fetch_lme_inventory_series(start, use_cache=use_cache)
    pmi = fetch_cn_pmi_series(start, use_cache=use_cache)

    feat_v5 = prepare_features_v5(df, wsr_d, hold_d, lme, inv, pmi)

    print(f"\n[calibrate v5] years={args.years}, step={args.step}, long_only=True ...")
    cal1 = run_calibration_backtest(feat_v5, min_train=args.min_train, step=args.step, horizon=1)
    cal5 = run_calibration_backtest(feat_v5, min_train=args.min_train, step=args.step, horizon=5)

    cfg = optimize_and_save_thresholds(cal1, cal5, long_only=True)
    save_config(cfg, version=5)
    print(f"\n[optimize] 阈值 -> {DATA_DIR / 'results' / 'model_config_v5.json'}")
    print(f"  5日做多阈值: ≥{cfg['filter']['5d']['th_up']*100:.0f}%, 子模型一致≥{cfg['filter']['5d']['min_agree']}")

    m1 = calibration_metrics(cal1, cfg["filter"]["1d"], long_only=True)
    m5 = _metrics_v5(cal5, cfg, long_only=True)
    print(format_calibration_report(m1, m5))
    if cfg.get("quality_filter"):
        q = cfg["quality_filter"]
        print(f"  质量过滤: RSI≤{q.get('max_rsi14')}, 20日位置≤{q.get('max_price_pos20', 0)*100:.0f}%")

    if args.audit:
        print("\n" + audit_report_v4_v5())

    if args.compare:
        feat_v4 = prepare_features_v4(df, wsr_d, hold_d, lme)
        cal5_v4 = run_cal_v4(feat_v4, min_train=args.min_train, step=args.step, horizon=5)
        cfg_v4 = load_config(version=4)
        m5_v4 = calibration_metrics(cal5_v4, cfg_v4["filter"]["5d"], long_only=True)
        print("\n" + "=" * 52)
        print("  v4 vs v5 对比（5日做多信号）")
        print("=" * 52)
        print(f"  v4 全样本准确率   : {m5_v4.get('accuracy')}%")
        print(f"  v4 做多信号准确率 : {m5_v4.get('filtered_accuracy')}%  (n={m5_v4.get('filtered_n_signals')})")
        print(f"  v5 全样本准确率   : {m5.get('accuracy')}%")
        print(f"  v5 做多信号准确率 : {m5.get('filtered_accuracy')}%  (n={m5.get('filtered_n_signals')})")
        print("  切换: .env 设 MODEL_VERSION=5 启用 v5；删除或改回 4 即回退")
        print("=" * 52)

    out_dir = DATA_DIR / "results"
    cal1.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_v5_1d.csv", index=False, encoding="utf-8-sig")
    cal5.drop(columns=["probs_dict"], errors="ignore").to_csv(out_dir / "calibration_v5_5d.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] 校准结果 -> {out_dir}")


if __name__ == "__main__":
    main()
