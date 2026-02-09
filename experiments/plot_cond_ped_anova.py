from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from experiments._plot import plot_results


if TYPE_CHECKING:
    from argparse import Namespace


def main(args: Namespace) -> None:
    with open(args.input_path, "rb") as f:
        data = pickle.load(f)

    for mode in ["normalized", "raw"]:
        fig = plot_results(
            res := [
                d
                for i, d in enumerate(data[f"results_{mode}"])
                if (not args.data_indices or i in args.data_indices)
            ],
            {
                "normalized": "HPI",
                "raw": "Unnormalized HPI",
            }[mode],
            titles=[
                rf"$\gamma = {q:.2f}$"
                for i, q in enumerate(data["region_quantiles"])
                if (not args.data_indices or i in args.data_indices)
            ],
            xlabel=r"$\gamma'$",
            xs=[
                d
                for i, d in enumerate(data["target_quantiles"])
                if (not args.data_indices or i in args.data_indices)
            ],
            figsize=(6.0 * len(res), args.height),
            share_yaxis=mode == "normalized",
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
            sort_results="key",
        )
        save_path = Path(args.output_dir) / f"{Path(args.input_path).stem}_{mode}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--data-indices", type=int, nargs="+")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--height", type=float, default=2.8)
    args = parser.parse_args()
    main(args)
