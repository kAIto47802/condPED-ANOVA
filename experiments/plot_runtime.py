from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments._plot import plot_results


if TYPE_CHECKING:
    from argparse import Namespace


def main(args: Namespace) -> None:
    data = np.load(args.input_path)
    shape = next(iter(data.values())).shape
    all_evaluators = list(set(k.split("|")[0] for k in data.keys()))
    all_trials = sorted(set(int(k.split("|")[1].replace("trials", "")) for k in data.keys()))
    evaluator_names = {
        "cond_ped_anova": "condPED-ANOVA (Ours)",
        "ped_anova_filtering": "PED-ANOVA w/ Filtering",
        "ped_anova_imputation": "PED-ANOVA w/ Imputation",
        "fanova_filtering": "f-ANOVA w/ Filtering",
        "fanova_imputation": "f-ANOVA w/ Imputation",
        "mdi_filtering": "MDI w/ Filtering",
        "mdi_imputation": "MDI w/ Imputation",
        "shap_filtering": "SHAP w/ Filtering",
        "shap_imputation": "SHAP w/ Imputation",
    }
    results = {
        display_name: np.vstack(
            [
                data.get(f"{evaluator}|{n_trials}trials", np.full(shape, np.nan))
                for n_trials in all_trials
            ]
        ).T
        for evaluator, display_name in evaluator_names.items()
        if evaluator in all_evaluators
    }
    fig = plot_results(
        [results],
        "Runtime / s",
        xs=[np.array(all_trials)],
        use_math_font_for_labels=False,
        logx=True,
        logy=True,
        xlabel="Number of Samples",
        figsize=(8.0, 5.4),
        colors={
            "condPED-ANOVA (Ours)": "#0072B2",
            "PED-ANOVA w/ Filtering": "#E69F00",
            "PED-ANOVA w/ Imputation": "#E69F00",
            "f-ANOVA w/ Filtering": "#009E73",
            "f-ANOVA w/ Imputation": "#009E73",
            "MDI w/ Filtering": "#CC79A7",
            "MDI w/ Imputation": "#CC79A7",
            "SHAP w/ Filtering": "#D55E00",
            "SHAP w/ Imputation": "#D55E00",
        },
        markers={
            "condPED-ANOVA (Ours)": "*",
            "PED-ANOVA w/ Filtering": "o",
            "PED-ANOVA w/ Imputation": "o",
            "f-ANOVA w/ Filtering": "^",
            "f-ANOVA w/ Imputation": "^",
            "MDI w/ Filtering": "D",
            "MDI w/ Imputation": "D",
            "SHAP w/ Filtering": "s",
            "SHAP w/ Imputation": "s",
        },
        markersizes={
            "condPED-ANOVA (Ours)": 11,
            "PED-ANOVA w/ Filtering": 8,
            "PED-ANOVA w/ Imputation": 8,
            "f-ANOVA w/ Filtering": 8,
            "f-ANOVA w/ Imputation": 8,
            "MDI w/ Filtering": 6,
            "MDI w/ Imputation": 6,
            "SHAP w/ Filtering": 6,
            "SHAP w/ Imputation": 6,
        },
        linestyles={
            "condPED-ANOVA (Ours)": "-",
            "PED-ANOVA w/ Filtering": "--",
            "PED-ANOVA w/ Imputation": ":",
            "f-ANOVA w/ Filtering": "--",
            "f-ANOVA w/ Imputation": ":",
            "MDI w/ Filtering": "--",
            "MDI w/ Imputation": ":",
            "SHAP w/ Filtering": "--",
            "SHAP w/ Imputation": ":",
        },
        legend_fontsize=10.5,
        legend_handlelength=3.0,
    )
    save_path = (Path(args.output_dir) / Path(args.input_path).stem).with_suffix(".pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {save_path}")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input .npz file containing runtimes.",
    )
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
