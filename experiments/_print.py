from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Literal


def print_table(
    filename: str | Path,
    data: list[list[Any]],
    mode: Literal["md", "csv"] = "md",
    float_format: str = "{:.4f}",
    tight: bool = False,
) -> None:
    data = [[float_format.format(x) if isinstance(x, float) else x for x in row] for row in data]
    formatted = {"md": lambda d: _format_md(d, tight=tight), "csv": _format_csv}[mode](data)
    Path(filename).write_text(formatted)


def _format_md(data: list[list[str]], tight: bool) -> str:
    bar_left = "|" if tight else "| "
    bar_right = "|" if tight else " |"
    bar_middle = "|" if tight else bar_right
    header_line = bar_left + bar_middle.join(data[0]) + bar_right
    separator_line = bar_left + bar_middle.join(["---"] * len(data[0])) + bar_right
    data_lines = [bar_left + bar_middle.join(row) + bar_right for row in data[1:]]
    return "\n".join([header_line, separator_line, *data_lines])


def _format_csv(data: list[list[str]]) -> str:
    data = [[x.replace('"', '""') for x in row] for row in data]
    data = [[f'"{x}"' if any(c in x for c in '",\n\r') else x for x in row] for row in data]
    return "\n".join([",".join(row) for row in data])
