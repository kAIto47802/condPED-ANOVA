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

    from optuna.distributions import BaseDistribution, CategoricalChoiceType
    from optuna.importance import BaseImportanceEvaluator
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _get_mean_value(
    dist: BaseDistribution,
) -> int | float | CategoricalChoiceType:
    if isinstance(dist, FloatDistribution):
        return (dist.low * dist.high) ** 0.5 if dist.log else (dist.low + dist.high) / 2
    elif isinstance(dist, IntDistribution):
        return int((dist.low * dist.high) ** 0.5 if dist.log else (dist.low + dist.high) / 2)
    elif isinstance(dist, CategoricalDistribution):
        return dist.choices[0]
    else:
        raise NotImplementedError(f"Unsupported distribution type: {type(dist)}")


def _get_distributions(
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
            else:
                assert distributions[name] == trial.distributions[name]
    return distributions


def get_evaluator_with_imputation(
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
            distributions = _get_distributions(study)
            for trial in study.trials:
                imputed_params = {
                    name: trial.params.get(name, _get_mean_value(distributions[name]))
                    for name in distributions.keys()
                }
                imputed_study.add_trial(
                    optuna.trial.create_trial(
                        params=imputed_params,
                        distributions=distributions,
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
