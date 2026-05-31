import os
import pandas as pd
import numpy as np
from scipy import stats
from evaluation.metrics import performance_summary


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.2f}%"


def _fmt_f(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


class ReportGenerator:
    def __init__(self, config):
        self.cfg = config

    def generate(
        self,
        factor_name: str,
        ic_summary: dict,
        ic_series: pd.Series,
        backtest_result: dict,
        ic_decay: pd.Series = None,
    ) -> str:
        lines = []
        sep = "=" * 65
        dash = "-" * 65

        lines += [
            sep,
            f"  单因子回测报告 — {factor_name}",
            sep,
            f"  测试区间 : {self.cfg.start_date} ~ {self.cfg.end_date}",
            f"  股票池   : {self.cfg.universe}",
            f"  调仓频率 : 月频",
            f"  交易成本 : 单边 {self.cfg.transaction_cost * 100:.2f}%",
            f"  行业中性 : {'是' if self.cfg.neutralize_industry else '否'}",
            f"  市值中性 : {'是' if self.cfg.neutralize_mktcap else '否'}",
            "",
        ]

        # ── IC Analysis ──────────────────────────────────────────
        lines += [dash, "  一、因子有效性（IC 分析）", dash]
        col_w = 16
        for k, v in ic_summary.items():
            val = _fmt_pct(v) if "比例" in k else _fmt_f(v, 4)
            lines.append(f"  {k:<{col_w}}: {val}")
        lines.append("")

        # ── Layered Performance ───────────────────────────────────
        lines += [dash, "  二、分层回测绩效（超额收益口径）", dash]
        g_rets = backtest_result["group_returns"]
        exc_rets = backtest_result["excess_returns"]
        labels = [k for k in g_rets if k != "LS"] + ["LS"]

        header = f"  {'':8s} {'年化收益':>8} {'年化波动':>8} {'夏普':>6} {'最大回撤':>8} {'月胜率':>6}"
        lines.append(header)
        lines.append("  " + "-" * 50)

        for lb in labels:
            ret_s = exc_rets.get(lb, g_rets[lb])
            pm = performance_summary(ret_s, self.cfg.risk_free_rate)
            ar = _fmt_pct(pm["年化收益率"])
            av = _fmt_pct(pm["年化波动率"])
            sh = _fmt_f(pm["夏普比率"], 2)
            md = _fmt_pct(pm["最大回撤"])
            wr = _fmt_pct(pm["月度胜率"])
            lines.append(f"  {lb:<8s} {ar:>8} {av:>8} {sh:>6} {md:>8} {wr:>6}")

        lines.append("")

        # ── Monotonicity ─────────────────────────────────────────
        lines += [dash, "  三、单调性检验", dash]
        group_labels = [k for k in g_rets if k not in ("LS",)]
        from evaluation.metrics import annual_return
        ar_vals = [annual_return(exc_rets[lb]) for lb in group_labels]
        valid = [(i + 1, v) for i, v in enumerate(ar_vals) if not np.isnan(v)]
        if len(valid) >= 3:
            xs, ys = zip(*valid)
            rho, p = stats.spearmanr(xs, ys)
            lines.append(f"  各层年化超额收益 Spearman ρ = {rho:.4f}  (p = {p:.4f})")
            verdict = "单调性强 ✓" if abs(rho) >= 0.8 else ("单调性中等" if abs(rho) >= 0.6 else "单调性弱")
            lines.append(f"  评估结论: {verdict}")
        lines.append("")

        # ── IC Decay ─────────────────────────────────────────────
        if ic_decay is not None:
            lines += [dash, "  四、IC 衰减（滞后1~12期）", dash]
            decay_row = "  " + "  ".join(
                [f"Lag{k}: {_fmt_f(v, 3)}" for k, v in ic_decay.items()]
            )
            lines.append(decay_row)
            lines.append("")

        # ── Conclusion ────────────────────────────────────────────
        lines += [dash, "  五、综合结论", dash]
        ic_mean = ic_summary.get("IC均值", 0)
        icir = ic_summary.get("ICIR", 0)
        ic_pos = ic_summary.get("IC>0比例", 0)
        p_val = ic_summary.get("p值", 1)

        score = 0
        hints = []
        if isinstance(ic_mean, float) and abs(ic_mean) >= 0.03:
            score += 1; hints.append(f"|IC均值|={abs(ic_mean):.4f} ≥ 0.03 ✓")
        else:
            hints.append(f"|IC均值|={abs(ic_mean):.4f} < 0.03 ✗")

        if isinstance(icir, float) and abs(icir) >= 0.5:
            score += 1; hints.append(f"|ICIR|={abs(icir):.4f} ≥ 0.5 ✓")
        else:
            hints.append(f"|ICIR|={abs(icir):.4f} < 0.5 ✗")

        if isinstance(ic_pos, float) and ic_pos >= 0.55:
            score += 1; hints.append(f"IC>0比例={ic_pos:.2%} ≥ 55% ✓")
        else:
            hints.append(f"IC>0比例={ic_pos:.2%} < 55% ✗")

        if isinstance(p_val, float) and p_val < 0.05:
            score += 1; hints.append(f"t检验 p={p_val:.4f} < 0.05（显著）✓")
        else:
            hints.append(f"t检验 p={p_val:.4f} ≥ 0.05（不显著）✗")

        verdict = {4: "优秀因子 ★★★", 3: "有效因子 ★★☆", 2: "弱有效因子 ★☆☆", 1: "无效因子 ☆☆☆"}.get(score, "无效因子 ☆☆☆")
        for h in hints:
            lines.append(f"  • {h}")
        lines += ["", f"  综合评级: {verdict}", sep, ""]

        report = "\n".join(lines)
        # save to file
        path = os.path.join(self.cfg.output_dir, f"report_{factor_name}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"  [report] saved → {path}")
        return report
