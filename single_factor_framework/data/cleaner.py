import pandas as pd
import numpy as np


def fill_price_gaps(price: pd.DataFrame, method: str = "ffill", limit: int = 5) -> pd.DataFrame:
    return price.fillna(method=method, limit=limit)


def remove_suspended(price: pd.DataFrame, volume: pd.DataFrame = None) -> pd.DataFrame:
    """Mark rows as NaN when daily change is exactly 0 for 3+ consecutive days."""
    ret = price.pct_change()
    consec_zero = (ret == 0).rolling(3).sum() >= 3
    cleaned = price.copy()
    cleaned[consec_zero] = np.nan
    return cleaned


def align_dates(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    return [df.loc[common] for df in dfs]
