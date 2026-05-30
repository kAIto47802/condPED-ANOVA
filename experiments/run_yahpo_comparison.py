from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr
from yahpo_gym import list_scenarios

from experiments._evaluator_registry import get_all_evaluator_names
from experiments._print import print_table


if TYPE_CHECKING:
    from argparse import Namespace


def _calc_corr(
    evaluator_names: list[str],
    scenario: str,
    instance: int,
    n_trials: int,
    output_dir: str,
) -> dict[str, float]:
    with open(
        Path(output_dir)
        / f"yahpo_{scenario}_{instance}_{evaluator_names[0]}_{n_trials}trials.stats_raw.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)
    data = {}
    for evaluator in evaluator_names:
        with open(
            Path(output_dir) / f"yahpo_{scenario}_{instance}_{evaluator}_{n_trials}trials.pkl",
            "rb",
        ) as f:
            data[evaluator] = pickle.load(f)["results_raw"]
    data = {e: {k: np.mean(v) for k, v in d.items()} for e, d in data.items()}
    means = {k: np.mean(v, axis=0)[2] for k, v in stats.items()}

    max_importances = {
        evaluator: {
            learner_id: max(
                {k: v for k, v in importances.items() if k.startswith(learner_id)}.values()
            )
            for learner_id in means
        }
        for evaluator, importances in data.items()
    }
    corr = {
        evaluator: pearsonr(list(val.values()), list(means.values()))[0]
        for evaluator, val in max_importances.items()
    }
    return corr


def main(args: Namespace) -> None:
    corrs = [
        _calc_corr(
            args.evaluator_names,
            args.scenario,
            instance,
            args.n_trials,
            args.output_dir,
        )
        for instance in args.instances
    ]
    evaluator_names = {
        "cond_ped_anova": "condPED-ANOVA (Ours)",
        "ped_anova_filtering": "PED-ANOVA w/ Filtering",
        "ped_anova_imputation": "PED-ANOVA w/ Imputation",
        "fanova_filtering": "f-ANOVA w/ Filtering",
        "fanova_imputation": "f-ANOVA w/ Imputation",
        "mdi_filtering": "MDI w/ Filtering",
        "mdi_imputation": "MDI w/ Imputation",
        "shap_filtering": "SHAP w/ Filtering",
        "shap_imputation": "SHAP w/ Imputation",
    }
    header = ["Instance ID"] + [evaluator_names[e] for e in args.evaluator_names]
    data = [
        [f"{instance}"] + [f"{corr[evaluator]:.2f}" for evaluator in args.evaluator_names]
        for instance, corr in zip(args.instances, corrs, strict=True)
    ]
    stats = ["Mean $\\pm$ StdErr"] + [
        f"{np.mean([corr[evaluator] for corr in corrs]):.2f} $\\pm$ "
        f"{np.std([corr[evaluator] for corr in corrs]) / np.sqrt(len(corrs)):.2f}"
        for evaluator in args.evaluator_names
    ]
    print_table(
        Path(args.output_dir) / f"yahpo_{args.scenario}_correlations_{args.n_trials}trials.md",
        [header, *data, stats],
        mode="md",
        tight=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="rbv2_super",
        choices=list_scenarios(),
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=[1053, 1457, 1063, 1479, 15, 1468],
    )
    parser.add_argument(
        "--evaluator-names",
        type=str,
        choices=get_all_evaluator_names(),
        nargs="+",
        default=[
            "cond_ped_anova",
            "ped_anova_filtering",
            "ped_anova_imputation",
            "fanova_filtering",
            "fanova_imputation",
            "mdi_filtering",
            "mdi_imputation",
            "shap_filtering",
            "shap_imputation",
        ],
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
