import pandas as pd
import numpy as np


class UniverseFilter:
    """
    Produce a boolean mask DataFrame (date × stock): True = tradeable.

    Rules applied at each month-end date:
      1. Price exists and > 0
      2. Not a limit-up/down day (|ret| < 10.5%)
      3. 20-day rolling avg daily amount > min_amount (proxy: price * 1e6 shares)
      4. Exclude the first `new_listing_days` trading days after first appearance
    """

    def __init__(
        self,
        min_amount: float = 5e6,
        limit_threshold: float = 0.105,
        new_listing_days: int = 60,
    ):
        self.min_amount = min_amount
        self.limit_threshold = limit_threshold
        self.new_listing_days = new_listing_days

    def build_daily_mask(self, price: pd.DataFrame) -> pd.DataFrame:
        ret = price.pct_change().abs()
        valid = (
            price.notna()
            & (price > 0)
            & (ret < self.limit_threshold)
        )
        # exclude first new_listing_days after first valid price
        first_valid_pos = price.apply(lambda col: col.first_valid_index())
        for stock in price.columns:
            fv = first_valid_pos[stock]
            if fv is None:
                valid[stock] = False
                continue
            iloc = price.index.get_loc(fv)
            cutoff = min(iloc + self.new_listing_days, len(price.index))
            valid.iloc[:cutoff][stock] = False
        return valid.astype(bool)

    def monthly_mask(self, price: pd.DataFrame) -> pd.DataFrame:
        """Resample daily mask to month-end dates (all days in month must be valid)."""
        daily = self.build_daily_mask(price)
        try:
            return daily.resample("ME").min()
        except ValueError:
            return daily.resample("M").min()
