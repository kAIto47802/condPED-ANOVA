from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from experiments._plot import plot_results


if TYPE_CHECKING:
    from argparse import Namespace


def main(args: Namespace) -> None:
    data = []
    for path in args.input_paths:
        with open(path, "rb") as f:
            data.append(pickle.load(f))

    for mode in ["normalized", "raw"]:
        fig = plot_results(
            [res[0] if isinstance(res := d[f"results_{mode}"], list) else res for d in data],
            {
                "normalized": "HPI",
                "raw": "Unnormalized HPI",
            }[mode],
            titles=args.titles,
            xlabel=r"$\gamma'$",
            xs=[d["target_quantiles"][0] if "target_quantiles" in d else None for d in data],
            figsize=(sum(6.0 if "target_quantiles" in d else 2.0 for d in data), args.height),
            share_yaxis=mode == "normalized",
            colors={
                "c": "#0072B2",
                "c0": "#0072B2",
                "c1": "#56B4E9",
                "x": "#CC79A7",
                "y": "#E69F00",
                "z": "#D55E00",
            },
            ecolors={
                "c": "#006097",
                "x": "#A46186",
                "y": "#B37B01",
            },
            linestyles={
                "c": "-",
                "c0": "-",
                "c1": ":",
                "x": "--",
                "y": "-.",
                "z": (0, (3, 1, 1, 1, 1, 1)),
            },
            hatches={
                "c": "xx",
                "x": "//",
                "y": "\\\\",
            },
            hatch_linewidths={
                "c": 0.03,
                "x": 1.0,
                "y": 1.0,
            },
            linewidths={
                "c": 2.1,
                "c0": 3.0,
                "c1": 3.0,
                "x": 3.0,
                "y": 3.0,
                "z": 3.0,
            },
            title_fontsize=15,
            legend_loc=args.legend_loc,
            legend_column=args.legend_column,
            legend_bbox_to_anchor=args.legend_bbox_to_anchor,
            y_tick_step=0.25 if mode == "normalized" else None,
            sort_results="key",
            font_scale=args.font_scale,
        )
        save_path = Path(args.output_dir) / f"{args.save_name}_{mode}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-name",
        type=str,
        default="baselines_activation-disjoint",
    )
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs="+",
        default=[
            "results/activation-disjoint_ped_anova_filtering_1000trials.pkl",
            "results/activation-disjoint_ped_anova_imputation_1000trials.pkl",
            "results/activation-disjoint_fanova_filtering_1000trials.pkl",
            "results/activation-disjoint_fanova_imputation_1000trials.pkl",
            "results/activation-disjoint_mdi_filtering_1000trials.pkl",
            "results/activation-disjoint_mdi_imputation_1000trials.pkl",
            "results/activation-disjoint_shap_filtering_1000trials.pkl",
            "results/activation-disjoint_shap_imputation_1000trials.pkl",
        ],
    )
    parser.add_argument(
        "--titles",
        type=str,
        nargs="*",
        default=[
            r"PED-ANOVA ($\gamma=1.0$) w/ Filtering",
            r"PED-ANOVA ($\gamma=1.0$) w/ Imputation",
            "f-ANOVA w/ Filtering",
            "f-ANOVA w/ Imputation",
            "MDI w/ Filtering",
            "MDI w/ Imputation",
            "SHAP w/ Filtering",
            "SHAP w/ Imputation",
        ],
    )
    parser.add_argument("--legend-loc", type=str, default="upper left")
    parser.add_argument("--legend-column", type=int, default=0)
    parser.add_argument("--legend-bbox-to-anchor", type=float, nargs="+", default=None)
    parser.add_argument("--height", type=float, default=2.8)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
