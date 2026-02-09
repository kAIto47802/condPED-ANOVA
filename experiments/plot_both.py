from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from experiments._plot import plot_results_multi_row


if TYPE_CHECKING:
    from argparse import Namespace


def main(args: Namespace) -> None:
    data = []
    for path in args.input_paths:
        with open(path, "rb") as f:
            data.append(pickle.load(f))

    heights_ratios = [0.95, 0.3]
    fig = plot_results_multi_row(
        [
            [res[0] if isinstance(res := d[f"results_{mode}"], list) else res for d in data]
            for mode in ["normalized", "raw"]
        ],
        args.ylabels,
        titles=args.titles,
        xlabel=r"$\gamma'$",
        xs=[[d["target_quantiles"][0] for d in data]] * 2,
        figsize=(6.0 * len(data), args.height * sum(heights_ratios)),
        share_xaxis=True,
        share_yaxis=[True, False],
        colors={
            "c": "#0072B2",
            "x": "#CC79A7",
            "y": "#E69F00",
        },
        linestyles={
            "c": "-",
            "x": "--",
            "y": "-.",
        },
        linewidths={
            "c": 2.1,
            "x": 3.0,
            "y": 3.0,
        },
        title_fontsize=15,
        legend_loc=args.legend_loc,
        legend_column=args.legend_column,
        legend_bbox_to_anchor=None,
        height_ratios=heights_ratios,
        y_tick_step=[0.25, None],
    )
    output_path = Path(args.output_dir) / f"{args.save_name}_both.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {output_path}")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-name",
        type=str,
        default="ablation_wo_regime_probabilities_1000trials",
    )
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs="+",
        default=[
            "results/activation-disjoint_wo_regime_probabilities_1000trials.pkl",
            "results/activation-overlap_wo_regime_probabilities_1000trials.pkl",
            "results/regime-dependent-domain_wo_regime_probabilities_1000trials.pkl",
        ],
    )
    parser.add_argument("--titles", type=str, nargs="*")
    parser.add_argument("--legend-loc", type=str, default="upper left")
    parser.add_argument("--legend-column", type=int, default=0)
    parser.add_argument("--ylabels", type=str, nargs="*", default=["HPI", "Unnormalized HPI"])
    parser.add_argument("--height", type=float, default=2.8)
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
