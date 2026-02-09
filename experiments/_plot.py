from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


if TYPE_CHECKING:
    from typing import Literal

    from matplotlib.figure import Figure


def plot_results(
    results: list[dict[str, np.ndarray]],
    ylabel: str,
    *,
    titles: list[str | None] | None = None,
    xlabel: str | None = None,
    xs: list[np.ndarray | None] | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    share_yaxis: bool = True,
    use_math_font_for_labels: bool = True,
    logx: bool = False,
    logy: bool = False,
    colors: dict[str, str] | str | None = None,
    markers: dict[str, str] | None = None,
    markersizes: dict[str, float] | None = None,
    ecolors: dict[str, str] | None = None,
    linestyles: dict[str, str | tuple[float, tuple[float, ...]]] | None = None,
    linewidths: dict[str, float] | None = None,
    hatches: dict[str, str] | None = None,
    hatch_linewidths: dict[str, float] | None = None,
    title_fontsize: int | float = 16,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_column: int | None = None,
    legend_fontsize: int | float = 15,
    legend_handlelength: int | float = 2.0,
    labelrotation: int | float = 0,
    xmargin: float | None = None,
    sort_results: Literal[
        "key",
        "value",
        "key:descending",
        "value:descending",
        "key:ascending",
        "key:ascending",
    ]
    | None = None,
    y_tick_step: float | None = None,
    x_tick_fontsize: int | float = 12,
    ax_position: tuple[float, float, float, float] | None = None,
    use_tight_layout: bool = True,
    font_scale: float = 1.0,
) -> Figure:
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(results),
        figsize=figsize,
        sharey=share_yaxis,
        gridspec_kw=dict(
            width_ratios=[
                3.0 if next(iter(result.values())).ndim == 2 else 1.0 for result in results
            ]
        ),
    )
    xs = xs or [None] * len(results)
    titles = titles or [None] * len(results)
    for i, (result, x, title) in enumerate(zip(results, xs, titles, strict=True)):
        ax_i = ax if len(results) == 1 else ax[i]
        colors_i = colors if isinstance(colors, dict) else {}.fromkeys(result.keys(), colors)
        results_i = (
            result
            if sort_results is None
            else dict(sorted(result.items()))
            if sort_results.startswith("key")
            else dict(sorted(result.items(), key=lambda item: np.mean(item[1])))  # type: ignore
        )
        if sort_results and sort_results.endswith(":descending"):
            results_i = dict(reversed(results_i.items()))
        for j, (name, values) in enumerate(results_i.items()):
            mean = np.mean(values, axis=0)
            stderr = np.std(values, axis=0) / np.sqrt(values.shape[0])
            if values.ndim == 1:
                ax_i.bar(
                    j,
                    mean,
                    yerr=stderr,
                    label=rf"${name}$" if use_math_font_for_labels else name,
                    color=(color := colors_i.get(name, None)),
                    alpha=0.6,
                    error_kw=dict(
                        ecolor=ecolors and ecolors.get(name, None),
                        capsize=8,
                        elinewidth=2,
                    ),
                )
                with mpl.rc_context(
                    {
                        "hatch.color": color,
                        "hatch.linewidth": (hatch_linewidths or {}).get(name, 1.5),
                    }
                ):
                    ax_i.bar(
                        j,
                        mean,
                        facecolor="none",
                        color=color,
                        hatch=hatches and hatches.get(name, None),
                    )
                if j == len(result) - 1:
                    ax_i.set_xticks(range(len(result)))
                    ax_i.set_xticklabels(
                        [rf"${n}$" if use_math_font_for_labels else n for n in results_i],
                        fontsize=x_tick_fontsize * font_scale,
                    )
                    if labelrotation:
                        ax_i.tick_params(axis="x", labelrotation=labelrotation)
                        for t in ax_i.get_xticklabels():
                            t.set_ha("right")
                    if xmargin is not None:
                        ax_i.set_xmargin(xmargin)
            elif values.ndim == 2:
                ax_i.plot(
                    x if x is not None else range(mean.shape[0]),
                    mean,
                    label=rf"${name}$" if use_math_font_for_labels else name,
                    color=colors_i.get(name, None),
                    marker=markers and markers.get(name, None),
                    markersize=markersizes and markersizes.get(name, None),
                    linestyle=linestyles and linestyles.get(name, None),
                    linewidth=linewidths and linewidths.get(name, None),
                )
                ax_i.fill_between(
                    x,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.3,
                    color=colors_i.get(name, None),
                )
                ax_i.set_xlabel(xlabel, fontsize=14 * font_scale)
            else:
                raise ValueError(f"Unsupported values ndim: {values.ndim}")

        if not i:
            ax_i.set_ylabel(ylabel, fontsize=14 * font_scale)
        if share_yaxis and i:
            ax_i.tick_params(axis="x", labelsize=15 * font_scale)
            ax_i.tick_params(axis="y", labelleft=False)
        else:
            ax_i.tick_params(axis="both", labelsize=15 * font_scale)
        if y_tick_step is not None:
            ax_i.yaxis.set_major_locator(MultipleLocator(y_tick_step))
        if i == (len(results) - 1 if legend_column is None else legend_column):
            ax_i.legend(
                fontsize=legend_fontsize * font_scale,
                labelspacing=0.05,
                handletextpad=0.4,
                handlelength=legend_handlelength,
                borderpad=0.3,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
            )
        ax_i.set_title(title, fontsize=title_fontsize * font_scale)

        ax_i.tick_params(axis="x" if i else "both", labelsize=15 * font_scale)
        ax_i.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        if logx:
            ax_i.set_xscale("log")
        if logy:
            ax_i.set_yscale("log")
        if ax_position:
            ax.set_position(ax_position)
    if use_tight_layout:
        fig.tight_layout()
    return fig


def plot_results_multi_row(
    results: list[list[dict[str, np.ndarray]]],
    ylabels: list[str],
    *,
    titles: list[list[str | None]] | None = None,
    xlabel: str | None = None,
    xs: list[list[np.ndarray | None]] | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    share_xaxis: bool = True,
    share_yaxis: list[bool] | None = None,
    use_math_font_for_labels: bool = True,
    logx: bool = False,
    logy: bool = False,
    colors: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
    linewidths: dict[str, float] | None = None,
    title_fontsize: int | float = 16,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_column: int | None = None,
    height_ratios: list[float] | None = None,
    y_tick_step: list[float | None] | None = None,
) -> Figure:
    assert all(len(row) == len(results[0]) for row in results), "Inconsistent number of columns."
    fig, ax = plt.subplots(
        nrows=len(results),
        ncols=len(results[0]),
        figsize=figsize,
        sharey=all(share_yaxis) if share_yaxis is not None else False,
        gridspec_kw=dict(height_ratios=height_ratios or [1.0] * len(results)),
    )
    xs = xs or [[None] * len(results[0])] * len(results)
    titles = titles or [[None] * len(results[0])] * len(results)
    for i, (result_row, xs_row, titles_row) in enumerate(zip(results, xs, titles, strict=True)):
        for j, (result, x, title) in enumerate(zip(result_row, xs_row, titles_row)):
            for name, values in sorted(result.items()):
                mean = np.mean(values, axis=0)
                stderr = np.std(values, axis=0) / np.sqrt(values.shape[0])
                ax[i, j].plot(
                    x if x is not None else range(mean.shape[0]),
                    mean,
                    label=rf"${name}$" if use_math_font_for_labels else name,
                    color=colors and colors.get(name, None),
                    linestyle=linestyles and linestyles.get(name, None),
                    linewidth=linewidths and linewidths.get(name, None),
                )
                ax[i, j].fill_between(
                    x,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.3,
                    color=colors and colors.get(name, None),
                )
                if not share_xaxis or i == (len(results) - 1):
                    ax[i, j].set_xlabel(xlabel, fontsize=14)

            if not j:
                ax[i, j].set_ylabel(ylabels[i], fontsize=14)
            if not share_xaxis or i == (len(results) - 1):
                ax[i, j].tick_params(axis="x", labelsize=15)
            else:
                ax[i, j].tick_params(axis="x", labelbottom=False)
            if share_yaxis is not None and share_yaxis[i] and j:
                ax[i, j].tick_params(axis="y", labelleft=False)
            else:
                ax[i, j].tick_params(axis="y", labelsize=15)
            if y_tick_step is not None and (s := y_tick_step[i]) is not None:
                ax[i, j].yaxis.set_major_locator(MultipleLocator(s))
            if not i and j == (len(result_row) - 1 if legend_column is None else legend_column):
                ax[i, j].legend(
                    fontsize=15,
                    labelspacing=0.05,
                    handletextpad=0.4,
                    borderpad=0.3,
                    loc=legend_loc,
                    bbox_to_anchor=legend_bbox_to_anchor,
                )
            ax[i, j].set_title(title, fontsize=title_fontsize)
            ax[i, j].grid(which="major", color="gray", linestyle="--", linewidth=0.5)
            if logx:
                ax[i, j].set_xscale("log")
            if logy:
                ax[i, j].set_yscale("log")
    if isinstance(share_yaxis, list):
        for i, share in enumerate(share_yaxis):
            if share:
                for ax_ij in ax[i, 1:]:
                    ax_ij.sharey(ax[i, 0])
    fig.tight_layout()
    return fig
