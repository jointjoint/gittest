from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class FactorTestConfig:
    start_date: str = "2019-01-01"
    end_date: str = "2023-12-31"
    universe: str = "SAMPLE"
    n_groups: int = 5
    weighting: str = "equal"           # equal | cap_weighted
    transaction_cost: float = 0.002    # one-way cost
    benchmark: str = "000300.SH"
    neutralize_industry: bool = True
    neutralize_mktcap: bool = True
    winsorize_method: str = "mad"      # mad | percentile
    winsorize_n: float = 5.0
    risk_free_rate: float = 0.025      # annualized
    min_stocks: int = 30               # min valid stocks per period
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = str(Path(__file__).parent / "output")
        os.makedirs(self.output_dir, exist_ok=True)
