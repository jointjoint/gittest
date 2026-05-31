from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings


class FactorPreprocessor:
    """
    Three-step cross-sectional preprocessing for each rebalance date:
      Step 1  Winsorize  (MAD or percentile)
      Step 2  Neutralize (industry dummies + log market-cap OLS residual)
      Step 3  Standardize (Z-score)
    """

    def __init__(self, config):
        self.cfg = config

    # ── public ────────────────────────────────────────────────────

    def process(
        self,
        factor: pd.DataFrame,          # date × stock raw factor
        industry: pd.Series,           # stock → industry label (static)
        mkt_cap: pd.DataFrame,         # date × stock market cap
        valid_mask: pd.DataFrame,      # date × stock bool mask
    ) -> pd.DataFrame:
        """Return processed factor DataFrame with same shape as input."""
        result = {}
        for dt in factor.index:
            raw = factor.loc[dt].copy()

            # apply universe mask
            if dt in valid_mask.index:
                mask = valid_mask.loc[dt].reindex(raw.index).fillna(False)
                raw = raw[mask]

            raw = raw.dropna()
            if len(raw) < self.cfg.min_stocks:
                continue

            # Step 1 – winsorize
            f = self._winsorize(raw)

            # Step 2 – neutralize
            ind = industry.reindex(f.index)
            cap = mkt_cap.loc[dt].reindex(f.index) if dt in mkt_cap.index else None
            if self.cfg.neutralize_industry or self.cfg.neutralize_mktcap:
                f = self._neutralize(f, ind, cap)

            # Step 3 – z-score
            std = f.std()
            if std > 0:
                f = (f - f.mean()) / std

            result[dt] = f

        return pd.DataFrame(result).T  # date × stock

    # ── private ───────────────────────────────────────────────────

    def _winsorize(self, f: pd.Series) -> pd.Series:
        if self.cfg.winsorize_method == "mad":
            med = f.median()
            mad = (f - med).abs().median()
            if mad == 0:
                return f.clip(f.quantile(0.01), f.quantile(0.99))
            n = self.cfg.winsorize_n
            return f.clip(med - n * mad, med + n * mad)
        # percentile fallback
        return f.clip(f.quantile(0.01), f.quantile(0.99))

    def _neutralize(
        self,
        f: pd.Series,
        industry: pd.Series,
        log_cap: pd.Series | None,
    ) -> pd.Series:
        controls = []
        if self.cfg.neutralize_industry and industry is not None:
            dummies = pd.get_dummies(industry, drop_first=True).astype(float)
            controls.append(dummies)
        if self.cfg.neutralize_mktcap and log_cap is not None:
            lc = np.log(log_cap.replace(0, np.nan)).rename("ln_cap")
            controls.append(lc.to_frame())

        if not controls:
            return f

        X = pd.concat(controls, axis=1).reindex(f.index).dropna(how="any")
        common = f.index.intersection(X.index)
        if len(common) < 10:
            return f

        X_c = sm.add_constant(X.loc[common], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resid = sm.OLS(f.loc[common], X_c).fit().resid
        return resid.reindex(f.index)
