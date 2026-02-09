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
    fig = plot_results(
        [data[f"results_{args.mode}"]],
        {
            "normalized": "HPI",
            "raw": "Unnormalized HPI",
        }[args.mode],
        figsize=(20, 3.2),
        colors="#0072B2",
        labelrotation=24,
        legend_column=-1,
        use_math_font_for_labels=False,
        xmargin=0.01,
        sort_results="value:descending",
        x_tick_fontsize=11,
        ax_position=(0.064, 0.43, 0.934, 0.56),
        use_tight_layout=False,
    )
    save_path = Path(args.output_dir) / f"{Path(args.input_path).stem}_{args.mode}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {save_path}")
    fig.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)
