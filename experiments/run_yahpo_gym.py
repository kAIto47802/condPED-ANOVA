from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import ConfigSpace as CS
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from yahpo_gym import BenchmarkSet, list_scenarios, local_config

from experiments._evaluator_registry import (
    get_all_evaluator_names,
    get_evaluator,
)


if TYPE_CHECKING:
    from argparse import Namespace

    import ConfigSpace.hyperparameters as CSH
    from optuna.distributions import BaseDistribution


def _run_experiment_once(
    args: Namespace, seed: int
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float, float]]]:
    b = BenchmarkSet(scenario=args.scenario, instance=args.instance)
    cs = b.get_opt_space(seed=seed)
    xs = cs.sample_configuration(size=args.n_trials)
    ys = b.objective_function(xs)
    dists = _convert_to_optuna_distributions(cs, exclude=["task_id"])

    study = optuna.create_study(direction="maximize")
    trials = [
        optuna.trial.create_trial(
            params=(p := {k: v for k, v in x.get_dictionary().items() if k != "task_id"}),
            distributions={k: dists[k] for k in p},
            value=y[args.metric],
        )
        for x, y in zip(xs, ys)
    ]
    study.add_trials(trials)

    trial_values = defaultdict(list)
    for t in study.trials:
        assert t.value is not None
        trial_values[t.params[args.stat_key]].append(t.value)
    stats = {
        learner_id: (min(vs), sum(vs) / len(vs), max(vs))
        for learner_id, vs in trial_values.items()
    }

    evaluator = get_evaluator(args.evaluator_name)()
    normalized_importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, normalize=False
    )
    return normalized_importances, importances, stats


def _convert_to_optuna_distributions(
    cs: CS.ConfigurationSpace, exclude: list[str] | None = None
) -> dict[str, BaseDistribution]:
    def _convert(hp: CSH.Hyperparameter) -> BaseDistribution:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            return FloatDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            return IntDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            return CategoricalDistribution(choices=hp.choices)
        else:
            raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

    return {
        name: _convert(hp) for name, hp in cs.items() if exclude is None or name not in exclude
    }


def main(args: Namespace) -> None:
    save_path = Path(args.output_dir) / (
        f"yahpo_{args.scenario}_{args.instance}_{args.evaluator_name.replace('/', '-')}"
        f"_{args.n_trials}trials.pkl"
    )
    if save_path.exists():
        print(f"Results already exist at {save_path}, skipping...")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)

    local_config.init_config()
    local_config.set_data_path(args.data_path)

    results_normalized_raw, results_raw, stats_raw = zip(
        *[_run_experiment_once(args, 42 + s) for s in range(args.n_seeds)]
    )
    results_normalized = {
        k: np.array([res[k] for res in results_normalized_raw]) for k in results_normalized_raw[0]
    }
    results = {k: np.array([res[k] for res in results_raw]) for k in results_raw[0]}

    stats = {k: np.array([res[k] for res in stats_raw]) for k in stats_raw[0]}
    means = {k: np.mean(v, axis=0) for k, v in stats.items()}
    stderrs = {k: np.std(v, axis=0) / np.sqrt(args.n_seeds) for k, v in stats.items()}

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "args": args,
                "results_normalized": results_normalized,
                "results_raw": results,
            },
            f,
        )
    with open(save_path.with_suffix(".stats.txt"), "w") as f:
        f.write("Learner ID\tMin\tMean\tMax\n")
        for learner_id in means:
            mean_min, mean_mean, mean_max = means[learner_id]
            stderr_min, stderr_mean, stderr_max = stderrs[learner_id]
            f.write(
                f"{learner_id}\t"
                f"{mean_min:.6f} ± {stderr_min:.6f}\t"
                f"{mean_mean:.6f} ± {stderr_mean:.6f}\t"
                f"{mean_max:.6f} ± {stderr_max:.6f}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="yahpo_data",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="rbv2_super",
        choices=list_scenarios(),
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="1053",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
    )
    parser.add_argument(
        "--stat-key",
        type=str,
        default="learner_id",
    )
    parser.add_argument(
        "--evaluator-name",
        type=str,
        choices=get_all_evaluator_names(),
        default="cond_ped_anova",
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
