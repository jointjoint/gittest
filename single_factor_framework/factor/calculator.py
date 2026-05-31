import pandas as pd
import numpy as np
from factor.base import BaseFactor


class ROEFactor(BaseFactor):
    """Return-on-Equity (TTM). Higher is better."""

    name = "ROE_TTM"
    direction = 1

    def compute(self, data: dict) -> pd.DataFrame:
        roe = data["roe"]  # date × stock, already monthly
        return roe.copy()


class MomentumFactor(BaseFactor):
    """
    12-1 momentum: cumulative return over past 12 months, skipping last month.
    Avoids short-term mean reversion.
    """

    name = "MOM_12_1"
    direction = 1

    def __init__(self, config, lookback: int = 12, skip: int = 1):
        super().__init__(config)
        self.lookback = lookback
        self.skip = skip

    def compute(self, data: dict) -> pd.DataFrame:
        price = data["price"]
        try:
            monthly = price.resample("ME").last()
        except ValueError:
            monthly = price.resample("M").last()

        ret_long = monthly.pct_change(self.lookback)
        ret_short = monthly.pct_change(self.skip)
        factor = ret_long - ret_short
        return factor.dropna(how="all")


class VolatilityFactor(BaseFactor):
    """
    Annualized realized volatility over past 20 trading days.
    Lower volatility stocks tend to outperform (low-vol anomaly).
    """

    name = "VOL_20D"
    direction = -1   # lower volatility → better

    def __init__(self, config, window: int = 20):
        super().__init__(config)
        self.window = window

    def compute(self, data: dict) -> pd.DataFrame:
        price = data["price"]
        daily_ret = price.pct_change()
        vol = daily_ret.rolling(self.window).std() * np.sqrt(252)
        try:
            monthly = vol.resample("ME").last()
        except ValueError:
            monthly = vol.resample("M").last()
        return monthly.dropna(how="all")


FACTOR_REGISTRY: dict[str, type[BaseFactor]] = {
    "ROE_TTM": ROEFactor,
    "MOM_12_1": MomentumFactor,
    "VOL_20D": VolatilityFactor,
}
