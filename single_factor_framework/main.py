"""
Single-Factor Backtesting Framework  —  Entry Point

Usage (run from the single_factor_framework directory):
    python main.py
    python main.py --factor ROE_TTM
    python main.py --factor MOM_12_1 --no-neutralize
"""

import sys
import argparse
from pathlib import Path

# make all submodules importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import FactorTestConfig
from data.loader import DataLoader
from data.universe import UniverseFilter
from factor.preprocessor import FactorPreprocessor
from factor.calculator import FACTOR_REGISTRY
from backtest.layered import LayeredBacktest
from evaluation.ic_analysis import ICAnalysis
from evaluation.metrics import performance_summary
from report.plotter import Plotter
from report.generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Single-Factor Backtest")
    parser.add_argument("--factor", default="ROE_TTM",
                        choices=list(FACTOR_REGISTRY.keys()),
                        help="Factor to test (default: ROE_TTM)")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2023-12-31")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost",   type=float, default=0.002)
    parser.add_argument("--no-neutralize", action="store_true")
    return parser.parse_args()


def run(factor_name: str, config: FactorTestConfig) -> dict:
    print(f"\n{'='*55}")
    print(f"  单因子回测系统  |  因子: {factor_name}")
    print(f"{'='*55}")

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    loader = DataLoader(config)
    price     = loader.load_price()
    roe       = loader.load_roe()
    industry  = loader.load_industry()
    mkt_cap   = loader.load_mkt_cap()
    benchmark = loader.load_benchmark()
    print(f"      price: {price.shape}  |  stocks: {len(price.columns)}")

    # ── 2. Month-end prices & returns ───────────────────────────
    print("[2/6] Computing monthly prices & universe filter...")
    # 取月末最后一个交易日收盘价，避免用月内均价引入未来信息
    monthly_price = price.resample("M").last()

    universe_filter = UniverseFilter()
    valid_mask = universe_filter.monthly_mask(price)

    # ── 3. Compute raw factor ────────────────────────────────────
    print(f"[3/6] Computing raw factor: {factor_name}...")
    FactorClass = FACTOR_REGISTRY[factor_name]
    factor_obj  = FactorClass(config)
    data_bundle = {
        "price":     price,
        "roe":       roe,
        "industry":  industry,
        "mkt_cap":   mkt_cap,
        "benchmark": benchmark,
    }
    raw_factor = factor_obj.compute(data_bundle)
    # 只保留 price 中存在的股票，剔除因子有值但价格缺失的标的
    raw_factor = raw_factor.reindex(columns=monthly_price.columns)
    # 因子日历（财报披露日）与价格日历可能不对齐，取交集避免 NaN 截面
    common_dates = raw_factor.index.intersection(monthly_price.index)
    raw_factor    = raw_factor.loc[common_dates]
    monthly_price = monthly_price.loc[common_dates]
    print(f"      factor shape: {raw_factor.shape}  |  dates: {len(common_dates)}")

    # ── 4. Preprocess factor ─────────────────────────────────────
    print("[4/6] Preprocessing factor (winsorize → neutralize → standardize)...")
    # 市值需要在因子对齐之后单独 resample，确保日期索引与 proc_factor 一致
    mkt_cap_monthly = mkt_cap.resample("M").last()

    preprocessor = FactorPreprocessor(config)
    proc_factor = preprocessor.process(
        raw_factor,
        industry,
        mkt_cap_monthly.reindex(columns=monthly_price.columns),
        valid_mask,
    )
    print(f"      processed dates: {len(proc_factor)}")

    # ── 5. IC Analysis ───────────────────────────────────────────
    print("[5/6] IC analysis...")
    monthly_returns = monthly_price.pct_change()
    ic_analyzer = ICAnalysis()
    ic_series   = ic_analyzer.compute(proc_factor, monthly_returns)
    ic_summary  = ic_analyzer.summary(ic_series)
    ic_decay    = ic_analyzer.decay(proc_factor, monthly_returns, lags=12)

    print(f"      IC Mean={ic_summary.get('IC均值', 'N/A')}  "
          f"ICIR={ic_summary.get('ICIR', 'N/A')}  "
          f"IC>0={ic_summary.get('IC>0比例', 'N/A')}")

    # ── 6. Layered backtest ──────────────────────────────────────
    print("[6/6] Running layered backtest...")
    backtester = LayeredBacktest(config)
    bt_result  = backtester.run(proc_factor, monthly_price, benchmark, mkt_cap_monthly)
    print(f"      backtest periods: {len(list(bt_result['group_returns'].values())[0])}")

    # ── Output ───────────────────────────────────────────────────
    print("\n  Generating charts & report...")
    plotter = Plotter(config)
    plotter.plot_all(bt_result, ic_series, ic_decay, factor_name)

    generator = ReportGenerator(config)
    report = generator.generate(factor_name, ic_summary, ic_series, bt_result, ic_decay)
    print()
    print(report)

    return {
        "config":       config,
        "factor_name":  factor_name,
        "ic_series":    ic_series,
        "ic_summary":   ic_summary,
        "ic_decay":     ic_decay,
        "bt_result":    bt_result,
        "proc_factor":  proc_factor,
    }


def main():
    args = parse_args()

    config = FactorTestConfig(
        start_date=args.start,
        end_date=args.end,
        n_groups=args.groups,
        transaction_cost=args.cost,
        neutralize_industry=not args.no_neutralize,
        neutralize_mktcap=not args.no_neutralize,
    )

    results = []
    # 用列表包装，方便后续扩展为多因子批量测试
    for fname in [args.factor]:
        result = run(fname, config)
        results.append(result)

    return results


if __name__ == "__main__":
    main()
