from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import cast, TYPE_CHECKING

import numpy as np
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.importance._base import _sort_dict_by_importance
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class CondPedAnovaImportanceEvaluator(PedAnovaImportanceEvaluator):
    """condPED-ANOVA importance evaluator.

    Implements the condPED-ANOVA hyperparameter importance evaluation algorithm.

    condPED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials
    better than a user-specified `target_quantile`.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than `target_quantile`.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`__

    `target_quantile` and `region_quantile` correspond to the parameters ``gamma'`` and ``gamma``
    in the original paper, respectively.

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above
        `target_quantile`. To stabilize the analysis, it is preferable to include at least
        5 trials above `target_quantile`.

    Args:
        target_quantile:
            Compute the importance of achieving top-``target_quantile`` quantile objective value.
            For example, ``target_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

        region_quantile:
            Define the region where we compute the importance. For example,
            ``region_quantile=0.5`` means that we compute the importance in the region where
            trials achieve top-50% performance. If ``region_quantile=1.0``, the importance is
            computed in the whole search space.

        baseline_quantile:
            Compute the importance of achieving top-``baseline_quantile`` quantile objective value.
            For example, ``baseline_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

            .. warning::
                Deprecated in v4.7.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v0.6.0, but this schedule is subject to
                change. `baseline_quantile` is currently ignored. Use `target_quantile` instead.
                See https://github.com/optuna/optuna/releases/tag/v4.7.0.

        evaluate_on_local:
            Whether we measure the importance in the local or global space.
            If :obj:`True`, the importances imply how importance each parameter is during
            optimization. Meanwhile, ``evaluate_on_local=False`` gives the importances in the
            specified search_space. ``evaluate_on_local=True`` is especially useful when users
            modify search space during optimization.

    Example:
        An example of using PED-ANOVA is as follows:

        .. testcode::

            import optuna

            from cond_ped_anova import CondPedAnovaImportanceEvaluator


            def objective(trial: optuna.trial.Trial) -> float:
                c = trial.suggest_float("c", 0, 1)
                if c < 0.5:
                    x = trial.suggest_float("x", -5, 3)
                    return x
                else:
                    y = trial.suggest_float("y", -3, 5)
                    return y

            sampler = optuna.samplers.RandomSampler(seed=42)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=100)
            evaluator = CondPedAnovaImportanceEvaluator()
            importance = optuna.importance.get_param_importances(study, evaluator=evaluator)

    """

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        dists = _get_distributions(study, params=params)
        if params is None:
            params = list({k for d in dists for k in d})

        assert params is not None

        trials = _get_filtered_trials(study, target=target)
        # The following should be tested at _get_filtered_trials.
        assert target is not None or max([len(t.values) for t in trials], default=1) == 1
        if len(trials) <= self._min_n_top_trials:
            return {k: 0.0 for k in params}

        target_trials = self._get_top_quantile_trials(study, trials, self._target_quantile, target)
        region_trials = (
            trials
            if self._region_quantile == 1.0
            else self._get_top_quantile_trials(study, trials, self._region_quantile, target)
        )
        quantile = len(target_trials) / len(region_trials)  # gamma' / gamma
        param_importances: dict[str, float] = defaultdict(float)
        for param_name in params:
            regime_trials = _partition_by_regime(param_name, region_trials)
            for dist, region_trials_regime in regime_trials.items():
                all_region_trials_regime = set(t._trial_id for t in region_trials_regime)
                target_trials_regime = [
                    t for t in target_trials if t._trial_id in all_region_trials_regime
                ]
                regime_prob_target = len(target_trials_regime) / len(target_trials)  # alpha_i
                regime_prob_region = len(region_trials_regime) / len(region_trials)  # beta_i
                if dist is not None and not dist.single() and len(target_trials_regime):
                    param_importances[param_name] += (
                        regime_prob_target**2
                        / regime_prob_region
                        * self._compute_pearson_divergence(
                            param_name,
                            dist,
                            target_trials=target_trials_regime,
                            region_trials=region_trials_regime,
                        )
                    )
                else:
                    param_importances[param_name] += 0.0
        param_importances = {k: v * quantile**2 for k, v in param_importances.items()}
        return _sort_dict_by_importance(param_importances)


def _partition_by_regime(
    param_name: str, trials: list[FrozenTrial]
) -> dict[BaseDistribution | None, list[FrozenTrial]]:
    # None for the inactive regime
    regime_trials: dict[BaseDistribution | None, list[FrozenTrial]] = defaultdict(list)
    for trial in trials:
        regime_trials[trial.distributions.get(param_name)].append(trial)

    return regime_trials


def _get_filtered_trials(
    study: Study, target: Callable[[FrozenTrial], float] | None
) -> list[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        trial
        for trial in trials
        if np.isfinite(
            target(trial) if target is not None else cast("float", trial.value)
        )  # TC006
    ]


def _get_distributions(
    study: Study, params: list[str] | None
) -> list[dict[str, BaseDistribution]]:
    if params is not None:
        raise NotImplementedError()
    trials = study.get_trials(deepcopy=False)
    return [
        t.distributions
        for t in trials
        if t.state
        in (
            TrialState.COMPLETE,
            TrialState.WAITING,
            TrialState.RUNNING,
        )
    ]
