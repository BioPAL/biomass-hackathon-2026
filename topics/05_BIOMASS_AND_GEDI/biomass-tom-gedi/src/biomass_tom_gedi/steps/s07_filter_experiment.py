"""Step 07 - does removing bright-at-all-heights TomoSAR profiles improve the TOM -> GEDI relationship?

For each filter (HV only / any pol; floor_db, top_db, mean_db; worst 1/2.5/5/10 % per pol) the shots are removed,
cells are rebuilt (cells with < min_shots shots drop out) and the ridge CV is repeated. Random removal of the same
number of shots is the control. Outputs in results/filter_experiment/.
"""
import logging

from ..config import Config
from ..dataset import Dataset
from ..experiment import run_filter_experiment, summarise_vs_baseline
from ..plotting import delta_r2_bar

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    ds = Dataset.from_config(cfg)
    out = cfg.results_dir / "filter_experiment"
    out.mkdir(exist_ok=True)
    fc = cfg.get("filter_experiment", default={}) or {}
    results, _ = run_filter_experiment(
        ds, quantiles=tuple(fc.get("quantiles", (0.01, 0.025, 0.05, 0.10))), modes=tuple(fc.get("modes", ("hv", "any"))),
        n_random=int(fc.get("n_random", 2)), levels=tuple(fc.get("levels", ("shot", "cell"))), out_dir=out)
    summary = summarise_vs_baseline(results)
    summary.round(4).to_csv(out / "summary_vs_baseline.csv", index=False)
    for level in summary.level.unique():
        delta_r2_bar(summary, level, out / f"dR2_vs_baseline_{level}.png")
    log.info("best per level:\n%s", summary.groupby("level").head(3)[["level", "config", "n", "r", "R2", "dR2_vs_base"]].round(3).to_string(index=False))
