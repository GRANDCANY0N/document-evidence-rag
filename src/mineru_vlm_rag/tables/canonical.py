from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from lxml import html


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def _comparison_clean(value: str) -> str:
    """Normalize representation only; keep every factual character/digit."""
    normalized = _clean(value)
    normalized = normalized.replace("\\%", "%")
    normalized = re.sub(r"\^\{([^{}]*)\}", r"\1", normalized)
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"[•●▪◦]", "", normalized)
    return re.sub(r"\s+", "", normalized)


def _table_node(content: str):
    root = html.fromstring(content)
    if root.tag.lower() == "table":
        return root
    tables = root.xpath(".//table")
    if not tables:
        raise ValueError("content does not contain a table")
    return tables[0]


@dataclass(frozen=True)
class CanonicalTable:
    grid: list[list[str]]
    header_rows: int
    source_pages: list[int | None]

    @property
    def width(self) -> int:
        return max((len(row) for row in self.grid), default=0)

    @property
    def height(self) -> int:
        return len(self.grid)

    def normalized_grid(self) -> list[list[str]]:
        return [[_clean(cell) for cell in row] for row in self.grid]


def canonicalize_table(content: str) -> CanonicalTable:
    """Expand rowspan/colspan so every logical cell has a stable coordinate."""
    table = _table_node(content)
    row_nodes = table.xpath(".//tr")
    if not row_nodes:
        raise ValueError("table has no rows")
    thead_rows = set(table.xpath("./thead/tr|.//thead/tr"))
    pending: dict[tuple[int, int], str] = {}
    grid: list[list[str]] = []
    source_pages: list[int | None] = []
    inferred_header_rows = 0
    has_explicit_thead = bool(thead_rows)

    for row_index, row in enumerate(row_nodes):
        output: list[str] = []
        column = 0

        def consume_pending() -> None:
            nonlocal column
            while (row_index, column) in pending:
                output.append(pending[(row_index, column)])
                column += 1

        consume_pending()
        cells = row.xpath("./th|./td")
        if row in thead_rows or (row_index == 0 and cells and all(cell.tag.lower() == "th" for cell in cells)):
            inferred_header_rows = row_index + 1
        elif row_index == 0 and not has_explicit_thead and any(
            cell.get("rowspan") or cell.get("colspan") for cell in cells
        ):
            # MinerU commonly emits multi-level headers with <td> rather than
            # <th>/<thead>. A span in the first row proves that the following
            # row is still part of the header.
            inferred_header_rows = min(2, len(row_nodes))
        for cell in cells:
            consume_pending()
            value = _clean(" ".join(cell.itertext()))
            try:
                rowspan = max(1, int(cell.get("rowspan", "1")))
                colspan = max(1, int(cell.get("colspan", "1")))
            except ValueError as exc:
                raise ValueError("rowspan/colspan must be integers") from exc
            for offset in range(colspan):
                output.append(value)
                for future_row in range(row_index + 1, row_index + rowspan):
                    pending[(future_row, column + offset)] = value
            column += colspan
        consume_pending()
        grid.append(output)
        raw_page = row.get("data-source-page")
        try:
            source_pages.append(int(raw_page) if raw_page else None)
        except ValueError:
            source_pages.append(None)

    width = max((len(row) for row in grid), default=0)
    if not width:
        raise ValueError("table has no cells")
    normalized = [row + [""] * (width - len(row)) for row in grid]
    if not has_explicit_thead and not inferred_header_rows:
        first = [_clean(value) for value in normalized[0]]
        header_terms = re.compile(
            r"^(?:项目|名称|类型|类别|年份|年度|日期|季度|指标|金额|数值|占比|比例|"
            r"资产|负债|险种|风险类型|压力情景|地区|机构|单位)(?:$|[（(])"
        )
        # A long narrative cell or numbered scenario is data, not a header.
        looks_like_data = any(len(value) > 40 for value in first) or bool(
            first and re.match(r"^(?:情景|场景)\s*\d+", first[0])
        )
        inferred_header_rows = 1 if not looks_like_data and any(header_terms.search(value) for value in first) else 0

    return CanonicalTable(
        grid=normalized,
        header_rows=min(inferred_header_rows, len(normalized)),
        source_pages=source_pages,
    )


def header_paths(table: CanonicalTable) -> list[str]:
    paths: list[str] = []
    for column in range(table.width):
        values: list[str] = []
        for row in range(table.header_rows):
            value = _clean(table.grid[row][column])
            if value and (not values or value != values[-1]):
                values.append(value)
        paths.append(" / ".join(values) or f"第{column + 1}列")
    counts: dict[str, int] = {}
    for value in paths:
        counts[value] = counts.get(value, 0) + 1
    return [
        f"{value} [第{index + 1}列]" if counts[value] > 1 else value
        for index, value in enumerate(paths)
    ]


def table_records(content: str) -> tuple[list[str], list[dict[str, Any]]]:
    table = canonicalize_table(content)
    headers = header_paths(table)
    records: list[dict[str, Any]] = []
    for row_index in range(table.header_rows, table.height):
        values = table.grid[row_index]
        if not any(_clean(value) for value in values):
            continue
        records.append(
            {
                "row_index": row_index - table.header_rows,
                "source_page": table.source_pages[row_index],
                "cells": {headers[index]: _clean(values[index]) for index in range(table.width)},
            }
        )
    return headers, records


def compare_table_html(left: str, right: str) -> dict[str, Any]:
    """Perform an exact, coordinate-aware comparison of two HTML tables."""
    try:
        left_table = canonicalize_table(left)
        right_table = canonicalize_table(right)
    except Exception as exc:
        return {"comparable": False, "agreed": False, "reason": type(exc).__name__, "conflicts": []}
    left_grid = left_table.normalized_grid()
    right_grid = right_table.normalized_grid()
    height = max(len(left_grid), len(right_grid))
    width = max(left_table.width, right_table.width)
    conflicts: list[dict[str, Any]] = []
    format_differences: list[dict[str, Any]] = []
    for row in range(height):
        for column in range(width):
            left_value = left_grid[row][column] if row < len(left_grid) and column < len(left_grid[row]) else None
            right_value = right_grid[row][column] if row < len(right_grid) and column < len(right_grid[row]) else None
            if left_value == right_value:
                continue
            if (
                left_value is not None
                and right_value is not None
                and _comparison_clean(left_value) == _comparison_clean(right_value)
            ):
                format_differences.append(
                    {"row": row, "column": column, "mineru": left_value, "vlm": right_value}
                )
            else:
                conflicts.append({"row": row, "column": column, "mineru": left_value, "vlm": right_value})
    return {
        "comparable": True,
        "agreed": not conflicts,
        "left_shape": [len(left_grid), left_table.width],
        "right_shape": [len(right_grid), right_table.width],
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:200],
        "conflicts_truncated": len(conflicts) > 200,
        "format_difference_count": len(format_differences),
        "format_differences": format_differences[:200],
        "format_differences_truncated": len(format_differences) > 200,
    }
