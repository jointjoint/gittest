"""
Generate synthetic stock market data for factor backtesting.

Design:
  - Each stock has a hidden quality parameter q ~ Uniform(0.2, 1.0)
  - ROE = 0.15 * q + N(0, 0.035)          → factor has real signal
  - daily_alpha = (q - 0.6) * 4e-4         → quality drives return
  - daily_return = beta*market + alpha + idio
"""

import numpy as np
import pandas as pd
from pathlib import Path


N_STOCKS = 100
N_INDUSTRIES = 10
SEED = 42


def generate_sample_data(
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
    n_stocks: int = N_STOCKS,
    seed: int = SEED,
    save_dir: str = None,
) -> dict:
    rng = np.random.default_rng(seed)

    # ── stock meta ────────────────────────────────────────────────
    codes = [f"{str(i + 1).zfill(6)}.SZ" if i < 50 else f"{str(i - 49).zfill(6)}.SH"
             for i in range(n_stocks)]

    ind_labels = [f"SW{str(k + 1).zfill(2)}" for k in range(N_INDUSTRIES)]
    industries = pd.Series(
        np.repeat(ind_labels, n_stocks // N_INDUSTRIES)[:n_stocks],
        index=codes, name="industry",
    )

    quality = rng.uniform(0.2, 1.0, n_stocks)   # hidden factor
    beta = rng.uniform(0.5, 1.5, n_stocks)
    init_price = rng.uniform(10.0, 100.0, n_stocks)
    log_shares = rng.uniform(np.log(5e7), np.log(5e9), n_stocks)  # float shares

    # ── trading calendar ─────────────────────────────────────────
    cal = pd.bdate_range(start_date, end_date)
    n_days = len(cal)

    # ── daily returns ────────────────────────────────────────────
    mkt_ret = rng.normal(3e-4, 0.012, n_days)

    stock_ret = np.zeros((n_days, n_stocks))
    daily_alpha = (quality - 0.6) * 4e-4
    for i in range(n_stocks):
        idio = rng.normal(0, 0.018, n_days)
        stock_ret[:, i] = beta[i] * mkt_ret + daily_alpha[i] + idio

    # ── price path ───────────────────────────────────────────────
    prices = np.empty((n_days, n_stocks))
    prices[0] = init_price
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * (1.0 + stock_ret[t])
    prices = np.clip(prices, 0.1, None)

    price_df = pd.DataFrame(prices, index=cal, columns=codes)

    # ── market cap ───────────────────────────────────────────────
    shares = np.exp(log_shares)
    mkt_cap_df = price_df.mul(shares, axis=1)

    # ── monthly financial data (ROE) ─────────────────────────────
    month_ends = pd.date_range(start_date, end_date, freq="ME")
    roe_rows = {}
    for dt in month_ends:
        roe_vals = quality * 0.15 + rng.normal(0, 0.035, n_stocks)
        roe_rows[dt] = roe_vals
    roe_df = pd.DataFrame(roe_rows, index=codes).T
    roe_df.index.name = "date"

    # ── benchmark ────────────────────────────────────────────────
    bm_ret = mkt_ret * 0.95 + rng.normal(0, 0.001, n_days)
    bm_nav = 3800.0 * np.cumprod(1.0 + bm_ret)
    benchmark_s = pd.Series(bm_nav, index=cal, name="000300.SH")

    result = {
        "price": price_df,
        "roe": roe_df,
        "industry": industries,
        "mkt_cap": mkt_cap_df,
        "benchmark": benchmark_s,
    }

    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        price_df.to_csv(p / "price.csv")
        roe_df.to_csv(p / "roe.csv")
        industries.to_csv(p / "industry.csv")
        mkt_cap_df.to_csv(p / "mkt_cap.csv")
        benchmark_s.to_csv(p / "benchmark.csv")

    return result
