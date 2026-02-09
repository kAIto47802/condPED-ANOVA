from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Callable

    from optuna.importance import BaseImportanceEvaluator

_EVALUATOR_REGISTRY: dict[str, type[BaseImportanceEvaluator]] = {}


def register_evaluator(
    name: str,
) -> Callable[[type[BaseImportanceEvaluator]], type[BaseImportanceEvaluator]]:
    def decorator(
        cls: type[BaseImportanceEvaluator],
    ) -> type[BaseImportanceEvaluator]:
        _EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_evaluator(name: str) -> type[BaseImportanceEvaluator]:
    return _EVALUATOR_REGISTRY[name]


def get_all_evaluator_names() -> list[str]:
    return list(_EVALUATOR_REGISTRY.keys())
