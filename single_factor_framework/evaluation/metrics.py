import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 12   # monthly rebalancing


def calc_nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def annual_return(returns: pd.Series) -> float:
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    total = (1.0 + returns.dropna()).prod()
    return float(total ** (PERIODS_PER_YEAR / n) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.025) -> float:
    rf_period = rf_annual / PERIODS_PER_YEAR
    excess = returns - rf_period
    vol = excess.std()
    if vol == 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    nav = calc_nav(returns)
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    ar = annual_return(returns)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(ar / mdd)


def win_rate(returns: pd.Series) -> float:
    valid = returns.dropna()
    if len(valid) == 0:
        return np.nan
    return float((valid > 0).mean())


def performance_summary(returns: pd.Series, rf_annual: float = 0.025) -> dict:
    return {
        "年化收益率": annual_return(returns),
        "年化波动率": annual_volatility(returns),
        "夏普比率":   sharpe_ratio(returns, rf_annual),
        "最大回撤":   max_drawdown(returns),
        "卡玛比率":   calmar_ratio(returns),
        "月度胜率":   win_rate(returns),
    }
