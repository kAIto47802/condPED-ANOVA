from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState

from experiments.baselines._utils import get_all_param_names


if TYPE_CHECKING:
    from typing import Callable

    from optuna.distributions import BaseDistribution
    from optuna.importance import BaseImportanceEvaluator
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _get_expanded_distributions(
    study: Study,
) -> dict[str, BaseDistribution]:
    param_names = get_all_param_names(study)
    distributions = {}
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        for name in param_names:
            if name not in trial.distributions:
                continue
            if name not in distributions:
                distributions[name] = trial.distributions[name]
            elif isinstance(
                dist := trial.distributions[name], (IntDistribution, FloatDistribution)
            ):
                assert isinstance(d := distributions[name], (IntDistribution, FloatDistribution))
                d.low = min(d.low, dist.low)
                d.high = max(d.high, dist.high)
            elif isinstance(dist, CategoricalDistribution):
                assert isinstance(d := distributions[name], CategoricalDistribution)
                d.choices = list(set(d.choices) | set(dist.choices))  # type: ignore
            else:
                raise ValueError(f"Unsupported distribution type: {type(dist)}")
    return distributions


def get_evaluator_with_expansion(
    evaluator_cls: type[BaseImportanceEvaluator],
) -> type[BaseImportanceEvaluator]:
    class _Evaluator(evaluator_cls):  # type: ignore[valid-type,misc]
        def evaluate(
            self,
            study: Study,
            params: list[str] | None = None,
            *,
            target: Callable[[FrozenTrial], float] | None = None,
        ) -> dict[str, float]:
            imputed_study = optuna.study.create_study(direction=study.direction)
            distributions = _get_expanded_distributions(study)
            for trial in study.trials:
                imputed_study.add_trial(
                    optuna.trial.create_trial(
                        params=trial.params,
                        distributions={name: distributions[name] for name in trial.params.keys()},
                        value=trial.value,
                        state=trial.state,
                    )
                )
            return super().evaluate(
                imputed_study,
                params=params,
                target=target,
            )

    return _Evaluator
