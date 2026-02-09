from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import optuna

from experiments._evaluator_registry import (
    get_all_evaluator_names,
    get_evaluator,
)
from experiments._objectives import get_all_objective_names, get_objective


if TYPE_CHECKING:
    from argparse import Namespace


def _run_experiment_once(
    objective_name: str, evaluator_name: str, n_trials: int, seed: int
) -> float:
    objective = get_objective(objective_name)
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    evaluator = get_evaluator(evaluator_name)()
    start = time.monotonic()
    optuna.importance.get_param_importances(study, evaluator=evaluator)
    end = time.monotonic()
    return end - start


def _parse_trial_max_ks(trial_max_ks: list[str]) -> dict[str, int]:
    return {item.split(":")[0]: int(item.split(":")[1]) for item in trial_max_ks}


def main(args: Namespace) -> None:
    save_path = Path(args.output_dir) / f"{args.objective_name}_runtimes.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results = dict(np.load(save_path)) if save_path.exists() else {}

    trial_max_ks = _parse_trial_max_ks(args.trial_max_ks)
    for evaluator_name in args.evaluator_names:
        for k in range(9, trial_max_ks.get(evaluator_name, 9) + 1):
            n_trials = 1 << k
            name = f"{evaluator_name}|{n_trials}trials"
            if name in results:
                print(f"Results already exist for {name}, skipping...")
                continue
            results[name] = np.array(
                [
                    _run_experiment_once(args.objective_name, evaluator_name, n_trials, 42 + s)
                    for s in range(args.n_seeds)
                ]
            )
            np.savez_compressed(save_path, **results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective-name",
        type=str,
        choices=get_all_objective_names(),
        default="activation-disjoint",
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
    parser.add_argument(
        "--trial-max-ks",
        type=str,
        nargs="+",
        default=[
            "cond_ped_anova:17",
            "ped_anova_filtering:17",
            "ped_anova_imputation:17",
            "fanova_filtering:11",
            "fanova_imputation:11",
            "mdi_filtering:17",
            "mdi_imputation:17",
            "shap_filtering:15",
            "shap_imputation:15",
        ],
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
