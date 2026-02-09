from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.importance import PedAnovaImportanceEvaluator

from experiments._evaluator_registry import get_all_evaluator_names, get_evaluator
from experiments._objectives import get_all_objective_names, get_objective


if TYPE_CHECKING:
    from argparse import Namespace


def _get_importances(
    study: optuna.Study,
    evaluator_name: str,
    target_quantile: float,
    region_quantile: float,
) -> tuple[dict[str, float], dict[str, float]]:
    print(
        f"Computing importances with {evaluator_name}, "
        f"target_quantile={target_quantile}, "
        f"region_quantile={region_quantile}"
    )
    evaluator_cls = get_evaluator(evaluator_name)
    assert issubclass(evaluator_cls, PedAnovaImportanceEvaluator)
    evaluator = evaluator_cls(target_quantile=target_quantile, region_quantile=region_quantile)

    normalized_importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, normalize=False
    )
    return normalized_importances, importances


def _run_experiment_once(
    args: Namespace, seed: int
) -> tuple[list[dict[str, list[float]]], list[dict[str, list[float]]], list[np.ndarray]]:
    objective = get_objective(args.objective_name)
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    target_quantiles_all = [
        np.arange(args.target_quantile_step, region_quantile, args.target_quantile_step)
        for region_quantile in args.region_quantiles
    ]
    results = [
        [
            _get_importances(
                study,
                args.evaluator_name,
                target_quantile,
                region_quantile,
            )
            for target_quantile in target_quantiles
        ]
        for target_quantiles, region_quantile in zip(target_quantiles_all, args.region_quantiles)
    ]
    normalized_importances, importances = zip(*[zip(*res) for res in results])

    normalized_importances = [
        {k: [imp[k] for imp in imps] for k in imps[0]} for imps in normalized_importances
    ]
    importances = [{k: [imp[k] for imp in imps] for k in imps[0]} for imps in importances]

    return normalized_importances, importances, target_quantiles_all


def main(args: Namespace) -> None:
    save_path = Path(args.output_dir) / (
        f"{args.objective_name}_{args.evaluator_name.replace('/', '-')}_{args.n_trials}trials.pkl"
    )
    if save_path.exists():
        print(f"Results already exist at {save_path}, skipping...")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results_normalized_raw, results_raw, target_quantiles = zip(
        *[_run_experiment_once(args, 42 + s) for s in range(args.n_seeds)]
    )
    results_normalized = [
        {
            k: np.array([res[i][k] for res in results_normalized_raw])
            for k in results_normalized_raw[0][i]
        }
        for i in range(len(args.region_quantiles))
    ]
    results = [
        {k: np.array([res[i][k] for res in results_raw]) for k in results_raw[0][i]}
        for i in range(len(args.region_quantiles))
    ]

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "args": args,
                "results_normalized": results_normalized,
                "results_raw": results,
                "target_quantiles": target_quantiles[0],  # same for all seeds
                "region_quantiles": args.region_quantiles,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective-name",
        type=str,
        choices=get_all_objective_names(),
        default="activation-disjoint",
    )
    parser.add_argument(
        "--evaluator-name",
        type=str,
        choices=get_all_evaluator_names(),
        default="cond_ped_anova",
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--region-quantiles", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    parser.add_argument("--target-quantile-step", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
