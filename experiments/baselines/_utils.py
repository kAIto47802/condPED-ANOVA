from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optuna.study import Study


def get_all_param_names(study: Study) -> list[str]:
    param_names = set(param_name for trial in study.trials for param_name in trial.params.keys())
    return list(param_names)
