import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_sample.generate_sample import generate_sample_data


class DataLoader:
    def __init__(self, config):
        self.config = config
        self._raw: dict | None = None

    def _ensure_loaded(self) -> dict:
        if self._raw is None:
            self._raw = generate_sample_data(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
        return self._raw

    def load_price(self) -> pd.DataFrame:
        d = self._ensure_loaded()
        return d["price"].loc[self.config.start_date: self.config.end_date].copy()

    def load_roe(self) -> pd.DataFrame:
        d = self._ensure_loaded()
        return d["roe"].loc[self.config.start_date: self.config.end_date].copy()

    def load_industry(self) -> pd.Series:
        return self._ensure_loaded()["industry"].copy()

    def load_mkt_cap(self) -> pd.DataFrame:
        d = self._ensure_loaded()
        return d["mkt_cap"].loc[self.config.start_date: self.config.end_date].copy()

    def load_benchmark(self) -> pd.Series:
        d = self._ensure_loaded()
        return d["benchmark"].loc[self.config.start_date: self.config.end_date].copy()
