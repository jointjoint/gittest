import numpy as np
import pandas as pd
from scipy import stats


class ICAnalysis:
    """Compute Rank IC (Spearman) series and summary statistics."""

    def compute(
        self,
        factor: pd.DataFrame,          # date × stock processed factor
        forward_returns: pd.DataFrame, # date × stock forward monthly returns
        method: str = "rank",          # rank | pearson
    ) -> pd.Series:
        """
        For each date T: IC_T = corr(factor[T], forward_returns[T+1])
        Returns a Series indexed by T (the signal date).
        """
        dates = sorted(factor.index)
        ret_dates = sorted(forward_returns.index)
        ic_dict = {}

        for i, t in enumerate(dates[:-1]):
            # find the forward return date right after t
            future = [d for d in ret_dates if d > t]
            if not future:
                continue
            t_next = future[0]

            f = factor.loc[t].dropna()
            r = forward_returns.loc[t_next].dropna()

            common = f.index.intersection(r.index)
            if len(common) < 20:
                continue

            f_c, r_c = f[common].values, r[common].values

            if method == "rank":
                ic, _ = stats.spearmanr(f_c, r_c)
            else:
                ic, _ = stats.pearsonr(f_c, r_c)

            if not np.isnan(ic):
                ic_dict[t] = float(ic)

        return pd.Series(ic_dict).sort_index()

    def summary(self, ic: pd.Series) -> dict:
        clean = ic.dropna()
        if len(clean) == 0:
            return {}
        t_stat, p_val = stats.ttest_1samp(clean, 0)
        icir = clean.mean() / clean.std() if clean.std() != 0 else np.nan
        return {
            "IC均值":        round(clean.mean(), 4),
            "IC标准差":      round(clean.std(), 4),
            "ICIR":          round(icir, 4),
            "IC>0比例":      round((clean > 0).mean(), 4),
            "|IC|>0.02比例": round((clean.abs() > 0.02).mean(), 4),
            "t统计量":       round(t_stat, 4),
            "p值":           round(p_val, 4),
            "样本数":        len(clean),
        }

    def decay(self, factor: pd.DataFrame, forward_returns: pd.DataFrame,
               lags: int = 12) -> pd.Series:
        """IC decay: compute IC for lag=1,2,...,lags periods ahead."""
        ic_at_lag = {}
        dates = sorted(factor.index)
        ret_dates = sorted(forward_returns.index)

        for lag in range(1, lags + 1):
            ic_list = []
            for i, t in enumerate(dates):
                future = [d for d in ret_dates if d > t]
                if len(future) < lag:
                    continue
                t_lag = future[lag - 1]
                f = factor.loc[t].dropna()
                r = forward_returns.loc[t_lag].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 20:
                    continue
                ic, _ = stats.spearmanr(f[common].values, r[common].values)
                if not np.isnan(ic):
                    ic_list.append(ic)
            ic_at_lag[lag] = float(np.mean(ic_list)) if ic_list else np.nan

        return pd.Series(ic_at_lag)
