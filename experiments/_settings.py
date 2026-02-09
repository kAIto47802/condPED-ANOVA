from __future__ import annotations

import importlib.util
import shutil

from optuna.importance import (
    BaseImportanceEvaluator,
    FanovaImportanceEvaluator,
    MeanDecreaseImpurityImportanceEvaluator,
    PedAnovaImportanceEvaluator,
)

from cond_ped_anova import CondPedAnovaImportanceEvaluator
from experiments._evaluator_registry import (
    register_evaluator,
)
from experiments.baselines.expansion import get_evaluator_with_expansion
from experiments.baselines.filtering import get_evaluator_with_filtering
from experiments.baselines.imputation import get_evaluator_with_imputation


_evaluators: dict[str, type[BaseImportanceEvaluator]] = {
    "ped_anova": PedAnovaImportanceEvaluator,
    "fanova": FanovaImportanceEvaluator,
    "mdi": MeanDecreaseImpurityImportanceEvaluator,
}
if importlib.util.find_spec("optuna_integration") is not None:
    _evaluators["shap"] = importlib.import_module("optuna_integration").ShapleyImportanceEvaluator

for name, cls in _evaluators.items():
    register_evaluator(f"{name}")(cls)
    register_evaluator(f"{name}_filtering")(get_evaluator_with_filtering(cls))
    register_evaluator(f"{name}_imputation")(get_evaluator_with_imputation(cls))
    register_evaluator(f"{name}_expansion")(get_evaluator_with_expansion(cls))

register_evaluator("cond_ped_anova")(CondPedAnovaImportanceEvaluator)

if importlib.util.find_spec("matplotlib") is not None and shutil.which("latex") is not None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],
        }
    )
