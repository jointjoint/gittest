"""
Chart generation module.
Uses English axis labels to ensure cross-platform font compatibility.
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from evaluation.metrics import calc_nav


def _configure_font():
    system = platform.system()
    candidates = {
        "Windows": ["Microsoft YaHei", "SimHei"],
        "Darwin":  ["PingFang SC", "Heiti SC"],
        "Linux":   ["WenQuanYi Zen Hei", "Noto Sans CJK SC"],
    }.get(system, [])
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_font()
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class Plotter:
    def __init__(self, config):
        self.cfg = config
        self.out = config.output_dir

    def plot_all(
        self,
        backtest_result: dict,
        ic_series: pd.Series,
        ic_decay: pd.Series = None,
        factor_name: str = "Factor",
    ):
        self.plot_layered_nav(backtest_result, factor_name)
        self.plot_ic_series(ic_series, factor_name)
        self.plot_ic_distribution(ic_series, factor_name)
        self.plot_group_annual_returns(backtest_result, factor_name)
        if ic_decay is not None:
            self.plot_ic_decay(ic_decay, factor_name)

    # ── individual charts ────────────────────────────────────────

    def plot_layered_nav(self, result: dict, factor_name: str = "Factor"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1, ax2 = axes

        group_nav = result["group_nav"]
        bm_ret = result["benchmark_ret"]
        bm_nav = calc_nav(bm_ret.reindex(
            sorted({d for s in group_nav.values() for d in s.index})
        ).fillna(0))

        labels_main = [k for k in group_nav if k != "LS"]
        for i, lb in enumerate(labels_main):
            nav = group_nav[lb]
            ax1.plot(nav.index, nav.values, label=lb, color=COLORS[i], linewidth=1.2)
        ax1.plot(bm_nav.index, bm_nav.values, label="Benchmark",
                 color="black", linewidth=1.5, linestyle="--")
        ax1.set_title(f"{factor_name} – Layered Portfolio NAV", fontsize=13)
        ax1.set_ylabel("NAV")
        ax1.legend(ncol=4, fontsize=8)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax1.grid(alpha=0.3)

        ls_nav = group_nav.get("LS")
        if ls_nav is not None:
            ax2.plot(ls_nav.index, ls_nav.values, color="crimson",
                     linewidth=1.5, label="Long-Short (Q5-Q1)")
            ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax2.set_title("Long-Short Portfolio NAV", fontsize=13)
            ax2.set_ylabel("NAV")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

        fig.tight_layout()
        self._save(fig, "layered_nav.png")

    def plot_ic_series(self, ic: pd.Series, factor_name: str = "Factor"):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(ic.index, ic.values, color=np.where(ic.values > 0, "#4C9BE8", "#E84C4C"),
               alpha=0.7, width=20)
        roll = ic.rolling(12, min_periods=3).mean()
        ax.plot(roll.index, roll.values, color="navy", linewidth=1.5,
                label="12-period rolling mean")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{factor_name} – Rank IC Time Series", fontsize=13)
        ax.set_ylabel("Rank IC")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        mean_ic = ic.mean()
        ax.text(0.02, 0.92, f"Mean IC={mean_ic:.4f}", transform=ax.transAxes,
                fontsize=9, color="navy")
        fig.tight_layout()
        self._save(fig, "ic_series.png")

    def plot_ic_distribution(self, ic: pd.Series, factor_name: str = "Factor"):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(ic.dropna(), bins=25, kde=True, ax=ax, color="#4C9BE8")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(ic.mean(), color="red", linewidth=1.5,
                   linestyle="--", label=f"Mean={ic.mean():.4f}")
        ax.set_title(f"{factor_name} – IC Distribution", fontsize=13)
        ax.set_xlabel("Rank IC")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "ic_distribution.png")

    def plot_group_annual_returns(self, result: dict, factor_name: str = "Factor"):
        from evaluation.metrics import annual_return
        group_rets = result["group_returns"]
        labels = [k for k in group_rets if k != "LS"]
        excess = result["excess_returns"]
        ar = [annual_return(excess[lb]) * 100 for lb in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#E84C4C" if v < 0 else "#4C9BE8" for v in ar]
        bars = ax.bar(labels, ar, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, ar):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{factor_name} – Annualized Excess Return by Group", fontsize=13)
        ax.set_ylabel("Annualized Excess Return (%)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "group_annual_returns.png")

    def plot_ic_decay(self, ic_decay: pd.Series, factor_name: str = "Factor"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(ic_decay.index, ic_decay.values, color="#4C9BE8", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{factor_name} – IC Decay (Lag 1–{len(ic_decay)})", fontsize=13)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Mean Rank IC")
        ax.set_xticks(ic_decay.index)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "ic_decay.png")

    # ── helper ───────────────────────────────────────────────────

    def _save(self, fig, filename: str):
        path = os.path.join(self.out, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [chart] saved → {path}")
