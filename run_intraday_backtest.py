"""
盘中修正规则回测 — 沪铜 + 原油 1 日概率对比

用法:
  python run_intraday_backtest.py
  python run_intraday_backtest.py --years 5 --step 3
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from config import DATA_DIR
from config_oil import OIL_DATA_DIR
from fetcher import fetch_cu_daily, fetch_cu_holding_series, fetch_cu_wsr_series
from forecast_oil_v5 import compute_probs as oil_compute_probs
from forecast_oil_v5 import load_oil_data, prepare_features as prepare_oil_features
from forecast_v5 import compute_probs as cu_compute_probs
from forecast_v5 import prepare_features as prepare_cu_features
from fundamentals import aggregate_holding, aggregate_wsr
from intraday_backtest import format_report, run_intraday_backtest
from lme_ratio import fetch_lme_ratio_series
from macro_features import fetch_cn_pmi_series, fetch_lme_inventory_series


def parse_args():
    p = argparse.ArgumentParser(description="盘中修正 vs 纯v5 回测")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--min-train", type=int, default=252)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    start = (pd.Timestamp.today() - timedelta(days=365 * args.years)).strftime("%Y%m%d")
    use_cache = not args.refresh

    print(f"\n[intraday backtest] years={args.years} step={args.step} ...")

    df_cu = fetch_cu_daily(start_date=start, use_cache=use_cache)
    wsr = fetch_cu_wsr_series(start, use_cache=use_cache)
    hold = fetch_cu_holding_series(start, use_cache=use_cache)
    lme = fetch_lme_ratio_series(df_cu, start, use_cache=use_cache)
    inv = fetch_lme_inventory_series(start, use_cache=use_cache)
    pmi = fetch_cn_pmi_series(start, use_cache=use_cache)
    feat_cu = prepare_cu_features(df_cu, aggregate_wsr(wsr), aggregate_holding(hold), lme, inv, pmi)

    df_oil, wsr_o, hold_o, ratio, api, pmi_o = load_oil_data(start, use_cache=use_cache)
    feat_oil = prepare_oil_features(df_oil, wsr_o, hold_o, ratio, api, pmi_o)

    cal_cu, sum_cu = run_intraday_backtest(
        feat_cu, cu_compute_probs, "CU0", min_train=args.min_train, step=args.step
    )
    cal_oil, sum_oil = run_intraday_backtest(
        feat_oil, oil_compute_probs, "SC0", min_train=args.min_train, step=args.step
    )

    print(format_report("沪铜 CU", sum_cu))
    print(format_report("原油 SC", sum_oil))

    out = DATA_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    cal_cu.to_csv(out / "intraday_backtest_cu.csv", index=False, encoding="utf-8-sig")
    cal_oil.to_csv(OIL_DATA_DIR / "results" / "intraday_backtest_oil.csv", index=False, encoding="utf-8-sig")
    print(f"\n[save] -> {out / 'intraday_backtest_cu.csv'}")
    print(f"[save] -> {OIL_DATA_DIR / 'results' / 'intraday_backtest_oil.csv'}")


if __name__ == "__main__":
    main()
