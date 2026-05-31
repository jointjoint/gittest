from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    """
    All factors inherit from this class.

    compute() returns a wide DataFrame: index=rebalance_dates, columns=stock_codes.
    """

    name: str = "UNNAMED"
    direction: int = 1   # 1 = higher is better, -1 = lower is better

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute(self, data: dict) -> pd.DataFrame:
        """
        Args:
            data: dict with keys price, roe, industry, mkt_cap, benchmark
        Returns:
            pd.DataFrame  shape=(n_dates, n_stocks)
        """
        raise NotImplementedError
