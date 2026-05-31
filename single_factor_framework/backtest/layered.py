from __future__ import annotations
import numpy as np
import pandas as pd
from backtest.portfolio import calc_turnover, group_return, calc_nav


class LayeredBacktest:
    """
    Core single-factor layered backtesting engine.

    At each rebalance date T:
      1. Sort valid stocks by processed factor into N equal groups
      2. Hold from T to T+1 (next rebalance date)
      3. Apply transaction cost based on portfolio turnover
      4. Compute group return, excess return vs benchmark
    """

    def __init__(self, config):
        self.cfg = config
        self.n = config.n_groups
        self.labels = [f"Q{i + 1}" for i in range(config.n_groups)]

    def run(
        self,
        factor: pd.DataFrame,          # date × stock processed factor
        monthly_price: pd.DataFrame,   # date × stock month-end price
        benchmark: pd.Series,          # date → benchmark price
        mkt_cap: pd.DataFrame = None,  # date × stock market cap (optional)
    ) -> dict:
        """
        Returns:
            dict with keys:
              group_returns  – {label: pd.Series of monthly returns}
              group_nav      – {label: pd.Series of cumulative NAV}
              excess_returns – {label: pd.Series of excess returns vs benchmark}
              holdings       – {label: list[(date, stocks)]}
              benchmark_ret  – pd.Series of monthly benchmark returns
              turnover       – {label: pd.Series of turnover per period}
        """
        # align factor dates with available price dates
        factor_dates = sorted(factor.index)
        price_dates = monthly_price.index.sort_values()

        # benchmark monthly return
        bm_ret = benchmark.pct_change().dropna()
        bm_monthly = bm_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)

        group_rets = {lb: {} for lb in self.labels}
        group_rets["LS"] = {}
        prev_holdings = {lb: pd.Index([]) for lb in self.labels}
        turnover_dict = {lb: {} for lb in self.labels}
        holdings_log = {lb: [] for lb in self.labels}

        for i, t in enumerate(factor_dates[:-1]):
            t_next = factor_dates[i + 1]

            # factor values at t
            f_t = factor.loc[t].dropna()
            if len(f_t) < self.cfg.n_groups * 5:
                continue

            # period return: price(t_next) / price(t) - 1
            if t in monthly_price.index and t_next in monthly_price.index:
                period_ret = (
                    monthly_price.loc[t_next] / monthly_price.loc[t] - 1.0
                )
            else:
                continue

            # cap weights
            cap_t = mkt_cap.loc[t] if (mkt_cap is not None and t in mkt_cap.index) else None

            # assign groups using rank to handle ties
            ranked = f_t.rank(method="first")
            try:
                group_labels = pd.qcut(
                    ranked,
                    q=self.n,
                    labels=self.labels,
                    duplicates="drop",
                )
            except ValueError:
                continue

            for lb in self.labels:
                stocks = group_labels[group_labels == lb].index
                holdings_log[lb].append((t, stocks))

                # turnover
                to = calc_turnover(prev_holdings[lb], stocks)
                turnover_dict[lb][t] = to
                prev_holdings[lb] = stocks

                # group return with transaction cost deducted
                ret = group_return(stocks, period_ret, self.cfg.weighting, cap_t)
                cost = to * self.cfg.transaction_cost * 2.0
                group_rets[lb][t_next] = ret - cost

            # long-short: Q_N - Q_1 (direction adjusted)
            r_top = group_rets[self.labels[-1]][t_next]
            r_bot = group_rets[self.labels[0]][t_next]
            ls_ret = r_top - r_bot
            group_rets["LS"][t_next] = ls_ret

        # convert to Series
        g_ret_series = {lb: pd.Series(v).sort_index() for lb, v in group_rets.items()}
        g_nav = {lb: calc_nav(s) for lb, s in g_ret_series.items()}

        # excess return vs benchmark
        excess = {}
        for lb, s in g_ret_series.items():
            if lb == "LS":
                excess[lb] = s
                continue
            bm = bm_monthly.reindex(s.index).fillna(0)
            excess[lb] = s - bm

        return {
            "group_returns": g_ret_series,
            "group_nav": g_nav,
            "excess_returns": excess,
            "holdings": holdings_log,
            "benchmark_ret": bm_monthly,
            "turnover": {lb: pd.Series(v).sort_index() for lb, v in turnover_dict.items()},
        }
