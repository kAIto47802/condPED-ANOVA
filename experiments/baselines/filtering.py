from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
from optuna.importance._base import _sort_dict_by_importance

from experiments.baselines._utils import get_all_param_names


if TYPE_CHECKING:
    from typing import Callable

    from optuna.importance import BaseImportanceEvaluator
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def get_evaluator_with_filtering(
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
            importances = {}
            for name in get_all_param_names(study):
                filtered_study = optuna.study.create_study(direction=study.direction)
                for trial in study.trials:
                    if name in trial.params:
                        filtered_study.add_trial(trial)
                importance = super().evaluate(
                    filtered_study,
                )
                importances[name] = importance[name]
            return _sort_dict_by_importance(importances)

    return _Evaluator
