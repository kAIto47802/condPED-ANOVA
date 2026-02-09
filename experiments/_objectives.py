from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

import optuna


if TYPE_CHECKING:
    from collections.abc import Callable

_OBJECTIVE_REGISTRY = {}


class _ObjectiveFunction(Protocol):
    def __call__(self, trial: optuna.trial.Trial) -> float: ...


def register_objective(
    name: str,
) -> Callable[[_ObjectiveFunction], _ObjectiveFunction]:
    def decorator(
        fn: _ObjectiveFunction,
    ) -> _ObjectiveFunction:
        _OBJECTIVE_REGISTRY[name] = fn
        return fn

    return decorator


def get_objective(name: str) -> Callable[[optuna.trial.Trial], float]:
    return _OBJECTIVE_REGISTRY[name]


def get_all_objective_names() -> list[str]:
    return list(_OBJECTIVE_REGISTRY.keys())


@register_objective("activation-disjoint")
def objective(trial: optuna.trial.Trial) -> float:  # noqa: F811
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        x = trial.suggest_float("x", -5.0, -2.0)
        return x
    else:
        y = trial.suggest_float("y", 2.0, 5.0)
        return y


@register_objective("activation-overlap")  # type: ignore[no-redef]
def objective(trial: optuna.trial.Trial) -> float:  # noqa: F811
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        x = trial.suggest_float("x", -5.0, 2.0)
        return x
    else:
        y = trial.suggest_float("y", -2.0, 5.0)
        return y


@register_objective("regime-dependent-domain")  # type: ignore[no-redef]
def objective(trial: optuna.trial.Trial) -> float:  # noqa: F811
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        x = trial.suggest_float("x", -7.0, -2.0)
        y = trial.suggest_float("y", -5.0, -2.0)
        return x + y
    else:
        x = trial.suggest_float("x", 2.0, 7.0)
        y = trial.suggest_float("y", 2.0, 5.0)
        return x + y


@register_objective("nested-conditions")  # type: ignore[no-redef]
def objective(trial: optuna.trial.Trial) -> float:  # noqa: F811
    c0 = trial.suggest_float("c0", 0.0, 1.0)
    if c0 < 0.75:
        c1 = trial.suggest_float("c1", 0.0, 1.0)
        if c1 < 0.5:
            x = trial.suggest_float("x", -7.0, -4.0)
            return x
        else:
            y = trial.suggest_float("y", -3.0, 0.0)
            return y
    else:
        z = trial.suggest_float("z", 2.0, 5.0)
        return z


@register_objective("three-way-branching")  # type: ignore[no-redef]
def objective(trial: optuna.trial.Trial) -> float:  # noqa: F811
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 1 / 3:
        x = trial.suggest_float("x", -7.0, -4.0)
        return x
    elif c < 2 / 3:
        x = trial.suggest_float("x", -3.0, 0.0)
        return x
    else:
        y = trial.suggest_float("y", 4.0, 7.0)
        return y
