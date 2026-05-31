import pandas as pd
import numpy as np


def calc_turnover(
    old_stocks: pd.Index,
    new_stocks: pd.Index,
) -> float:
    """One-way turnover rate: fraction of portfolio value that changes."""
    if len(old_stocks) == 0 or len(new_stocks) == 0:
        return 1.0
    sold = len(set(old_stocks) - set(new_stocks))
    bought = len(set(new_stocks) - set(old_stocks))
    return (sold + bought) / (2.0 * max(len(old_stocks), len(new_stocks)))


def calc_nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def group_return(
    group_stocks: pd.Index,
    period_ret: pd.Series,
    weighting: str = "equal",
    cap: pd.Series = None,
) -> float:
    valid = period_ret.reindex(group_stocks).dropna()
    if len(valid) == 0:
        return np.nan
    if weighting == "cap_weighted" and cap is not None:
        w = cap.reindex(valid.index).fillna(0)
        w_sum = w.sum()
        if w_sum > 0:
            return float((valid * w / w_sum).sum())
    return float(valid.mean())
