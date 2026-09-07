from __future__ import annotations

import re
from statistics import median

from mineru_vlm_rag.domain.models import Block, BlockType, Page


TEMPLATE_TYPES = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
VISUAL_TYPES = {BlockType.TABLE, BlockType.IMAGE, BlockType.CHART, BlockType.FORMULA}


def _center_x(block: Block) -> float:
    assert block.bbox is not None
    return (block.bbox.x0 + block.bbox.x1) / 2


def _center_y(block: Block) -> float:
    assert block.bbox is not None
    return (block.bbox.y0 + block.bbox.y1) / 2


def _mineru_order(block: Block) -> int:
    value = block.metadata.get("mineru_order", block.reading_order)
    try:
        return int(value)
    except (TypeError, ValueError):
        return block.reading_order


def _coordinate_width(page: Page, blocks: list[Block]) -> float:
    explicit = page.metadata.get("mineru_coordinate_width")
    if explicit:
        return float(explicit)
    max_x = max(block.bbox.x1 for block in blocks if block.bbox)
    if page.width and max_x <= page.width * 1.5:
        return page.width
    return max_x or page.width or 1.0


def _clear_derived_layout(blocks: list[Block]) -> None:
    for block in blocks:
        block.metadata.pop("column_index", None)
        block.metadata.pop("column_count", None)
        block.metadata.pop("resolved_reading_order", None)
        block.metadata.pop("previous_block_id", None)
        block.metadata.pop("next_block_id", None)


def _column_order(blocks: list[Block], page_width: float) -> tuple[list[Block], bool]:
    """Use column-major order only when every inferred column has evidence.

    A single narrow label must not become its own column. That was the cause
    of the page-3 "执笔" block jumping ahead of "总纂/统稿".
    """
    if len(blocks) < 4:
        return sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0, _mineru_order(block))), False
    by_x = sorted(blocks, key=lambda block: (_center_x(block), block.bbox.y0, _mineru_order(block)))
    gaps = [
        (_center_x(right) - _center_x(left), index)
        for index, (left, right) in enumerate(zip(by_x, by_x[1:]))
    ]
    split_after = {index for gap, index in gaps if gap >= page_width * 0.14}
    groups: list[list[Block]] = []
    current: list[Block] = []
    for index, block in enumerate(by_x):
        current.append(block)
        if index in split_after:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    if not 2 <= len(groups) <= 3 or any(len(group) < 2 for group in groups):
        return sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0, _mineru_order(block))), False
    centers = [median(_center_x(block) for block in group) for group in groups]
    if min(right - left for left, right in zip(centers, centers[1:])) < page_width * 0.12:
        return sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0, _mineru_order(block))), False

    ordered: list[Block] = []
    for column_index, group in enumerate(groups):
        for block in sorted(group, key=lambda value: (value.bbox.y0, value.bbox.x0, _mineru_order(value))):
            block.issue_flags = list(dict.fromkeys([*block.issue_flags, "LAY-001"]))
            block.metadata["column_index"] = column_index
            block.metadata["column_count"] = len(groups)
            ordered.append(block)
    return ordered, True


def _context_rank(block: Block) -> tuple[int, float, float]:
    value = re.sub(r"\s+", "", block.content)
    if re.match(r"^(?:附表|表|图)\d+", value):
        rank = 0
    elif re.match(r"^[（(]?(?:单位|币种)[：:]", value):
        rank = 2
    elif (
        re.match(r"^[（(]?(?:注|说明|数据来源|资料来源)[：:]", value)
        or re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", value)
        or block.block_type == BlockType.FOOTNOTE
    ):
        rank = 4
    elif re.fullmatch(r"[（(].*(?:余额|日期).*[）)]", value):
        rank = 3
    else:
        rank = 1
    return rank, block.bbox.y0 if block.bbox else float("inf"), block.bbox.x0 if block.bbox else float("inf")


def _is_rotated_table_layout(positioned: list[Block]) -> bool:
    tables = [block for block in positioned if block.block_type == BlockType.TABLE and block.bbox]
    tall_tables = [
        block for block in positioned
        if block.block_type == BlockType.TABLE
        and block.bbox
        and block.bbox.height >= block.bbox.width * 1.6
    ]
    marginal_contexts = [
        block for block in positioned
        if block.block_type not in VISUAL_TYPES
        and block.bbox
        and block.bbox.height >= block.bbox.width * 1.8
    ]
    return len(tall_tables) >= 2 or bool(tables and len(marginal_contexts) >= 2)


def _rotated_table_order(positioned: list[Block]) -> list[Block]:
    """Order side-by-side, 90-degree statistical tables as logical groups."""
    anchors = sorted(
        [block for block in positioned if block.block_type == BlockType.TABLE],
        key=lambda block: (block.bbox.x0, block.bbox.y0),
    )
    by_anchor: dict[str, list[Block]] = {anchor.block_id: [] for anchor in anchors}
    ungrouped: list[Block] = []
    anchor_ids = set(by_anchor)
    for block in positioned:
        if block.block_id in anchor_ids:
            continue
        table_id = str(block.metadata.get("associated_table_block_id") or "")
        if table_id in by_anchor:
            by_anchor[table_id].append(block)
        else:
            ungrouped.append(block)

    ordered: list[Block] = []
    heading = [block for block in ungrouped if block.bbox and block.bbox.y1 < min(a.bbox.y0 for a in anchors)]
    ordered.extend(sorted(heading, key=lambda block: (block.bbox.y0, block.bbox.x0)))
    ungrouped = [block for block in ungrouped if block not in heading]
    for column_index, anchor in enumerate(anchors):
        contexts = sorted(by_anchor[anchor.block_id], key=_context_rank)
        before = [value for value in contexts if _context_rank(value)[0] < 4]
        after = [value for value in contexts if _context_rank(value)[0] >= 4]
        for value in [*before, anchor, *after]:
            value.issue_flags = list(dict.fromkeys([*value.issue_flags, "LAY-002"]))
            value.metadata["column_index"] = column_index
            value.metadata["column_count"] = len(anchors)
            value.metadata["orientation_degrees"] = 90
            ordered.append(value)
    ordered.extend(sorted(ungrouped, key=lambda block: (block.bbox.x0, block.bbox.y0, _mineru_order(block))))
    return ordered


_YEAR_PREFIX_RE = re.compile(
    r"(?:截至|自|至|到)\s*\d{4}\s*年(?:第?[一二三四1234]\s*季度)?\s*$"
)
_YEAR_CONTINUATION_RE = re.compile(r"^(?:末|初|底|以来|上半年|下半年)[，,、：:]?")


def _repair_local_continuations(blocks: list[Block]) -> tuple[list[Block], int]:
    """Repair a narrow two-column boundary inversion using textual evidence.

    Example: a left-column block ends with ``截至2024年`` while a right-column
    block just above it starts with ``末，``.  Geometry-only ordering can place
    the continuation first.  The strict prefix/continuation pair gives enough
    evidence to swap the two without asking a VLM to reorder the page.
    """
    ordered = list(blocks)
    repairs = 0
    for prefix_index, prefix in enumerate(list(ordered)):
        if not prefix.bbox or not _YEAR_PREFIX_RE.search(prefix.content.strip()):
            continue
        candidates = [
            (index, continuation)
            for index, continuation in enumerate(ordered[:prefix_index])
            if continuation.bbox
            and continuation.block_type == BlockType.TEXT
            and _YEAR_CONTINUATION_RE.match(continuation.content.strip())
            and prefix.bbox.x0 < continuation.bbox.x0
            and abs(prefix.bbox.y0 - continuation.bbox.y0) <= 100
        ]
        if not candidates:
            continue
        continuation_index, continuation = candidates[-1]
        ordered.pop(prefix_index)
        continuation_index = ordered.index(continuation)
        ordered.insert(continuation_index, prefix)
        prefix.metadata["reading_order_repair"] = "year_prefix_before_continuation"
        continuation.metadata["reading_order_repair"] = "year_continuation_after_prefix"
        repairs += 1
    return ordered, repairs


def rebuild_reading_order(page: Page) -> Page:
    """Derive a conservative reading order while preserving MinerU order."""
    for block in page.blocks:
        block.metadata.setdefault("mineru_order", block.reading_order)
    _clear_derived_layout(page.blocks)

    positioned = [block for block in page.blocks if block.bbox and block.block_type not in TEMPLATE_TYPES]
    unpositioned = [block for block in page.blocks if not block.bbox and block.block_type not in TEMPLATE_TYPES]
    templates = [block for block in page.blocks if block.block_type in TEMPLATE_TYPES]
    if not positioned:
        page.blocks = sorted(page.blocks, key=_mineru_order)
        page.metadata["reading_order_strategy"] = "mineru_fallback"
        page.metadata["reading_order_confidence"] = "low"
        return _finalize(page)

    if _is_rotated_table_layout(positioned):
        ordered = _rotated_table_order(positioned)
        page.metadata["reading_order_strategy"] = "rotated_table_groups"
        page.metadata["reading_order_confidence"] = "medium"
    else:
        page_width = _coordinate_width(page, positioned)
        marginal_vertical = [
            block for block in positioned
            if block.block_type == BlockType.TEXT
            and block.bbox.height >= block.bbox.width * 3
            and (block.bbox.x0 >= page_width * 0.80 or block.bbox.x1 <= page_width * 0.12)
            and not block.metadata.get("associated_table_block_id")
        ]
        marginal_ids = {block.block_id for block in marginal_vertical}
        context_by_visual: dict[str, list[Block]] = {}
        attached_ids: set[str] = set()
        for block in positioned:
            visual_id = str(block.metadata.get("associated_table_block_id") or "")
            if visual_id:
                context_by_visual.setdefault(visual_id, []).append(block)
                attached_ids.add(block.block_id)

        visual_separators = [
            block for block in positioned
            if block.block_id not in marginal_ids
            and block.block_type in VISUAL_TYPES and block.bbox.width >= page_width * 0.66
        ]
        full_width_text = [
            block for block in positioned
            if block.block_id not in marginal_ids
            and block.block_type not in VISUAL_TYPES
            and block.block_id not in attached_ids
            and block.bbox.width >= page_width * 0.72
        ]
        # A heading can span the usable two-column body while occupying only
        # about 60% of the normalized full page (wide page margins account for
        # the rest).  Treat only explicit heading-shaped text this way; the
        # narrower threshold must not promote ordinary prose to a separator.
        cross_column_headings = [
            block for block in positioned
            if block.block_id not in marginal_ids
            and block.block_type not in VISUAL_TYPES
            and block.block_id not in attached_ids
            and block.bbox.width >= page_width * 0.58
            and block.bbox.height <= page_width * 0.08
            and (
                block.block_type == BlockType.TITLE
                or block.metadata.get("text_level") is not None
                or re.match(r"^(?:第[一二三四五六七八九十百\d]+[编章节篇]|专栏|专题)", block.content.strip())
            )
        ]
        separators = list({
            block.block_id: block
            for block in [*visual_separators, *full_width_text, *cross_column_headings]
        }.values())

        def separator_start(block: Block) -> float:
            contexts = context_by_visual.get(block.block_id, [])
            return min([block.bbox.y0, *[value.bbox.y0 for value in contexts if value.bbox]])

        separators.sort(key=lambda block: (separator_start(block), block.bbox.x0, _mineru_order(block)))
        separator_ids = {block.block_id for block in separators}
        regional = [
            block for block in positioned
            if block.block_id not in separator_ids
            and block.block_id not in attached_ids
            and block.block_id not in marginal_ids
        ]
        ordered: list[Block] = []
        used_columns = False
        consumed: set[str] = set()
        cursor = float("-inf")
        for separator in separators:
            start = separator_start(separator)
            before = [
                block for block in regional
                if block.block_id not in consumed and cursor <= _center_y(block) < start
            ]
            region_order, region_columns = _column_order(before, page_width)
            ordered.extend(region_order)
            used_columns = used_columns or region_columns
            consumed.update(block.block_id for block in before)
            contexts = sorted(context_by_visual.get(separator.block_id, []), key=_context_rank)
            ordered.extend(contexts)
            ordered.append(separator)
            consumed.update(block.block_id for block in contexts)
            consumed.add(separator.block_id)
            cursor = max(cursor, separator.bbox.y1)
        remaining = [block for block in regional if block.block_id not in consumed]
        region_order, region_columns = _column_order(remaining, page_width)
        ordered.extend(region_order)
        used_columns = used_columns or region_columns
        for block in sorted(marginal_vertical, key=lambda value: (value.bbox.x0, value.bbox.y0)):
            block.metadata["orientation_degrees"] = 90
            block.metadata["reading_order_role"] = "marginal_vertical"
            ordered.append(block)
        seen = {value.block_id for value in ordered}
        missing = [block for block in positioned if block.block_id not in seen]
        ordered.extend(sorted(missing, key=lambda block: (block.bbox.y0, block.bbox.x0, _mineru_order(block))))
        page.metadata["reading_order_strategy"] = "layout_columns" if used_columns else "layout_rows"
        page.metadata["reading_order_confidence"] = "high" if used_columns else "medium"

    ordered, continuation_repairs = _repair_local_continuations(ordered)
    if continuation_repairs:
        page.metadata["reading_order_continuation_repairs"] = continuation_repairs
    else:
        page.metadata.pop("reading_order_continuation_repairs", None)
    ordered.extend(sorted(unpositioned, key=_mineru_order))
    ordered.extend(sorted(templates, key=_mineru_order))
    page.blocks = ordered
    return _finalize(page)


def _finalize(page: Page) -> Page:
    for index, block in enumerate(page.blocks):
        block.reading_order = index
        block.metadata["resolved_reading_order"] = index
        block.metadata["previous_block_id"] = page.blocks[index - 1].block_id if index else None
        block.metadata["next_block_id"] = page.blocks[index + 1].block_id if index + 1 < len(page.blocks) else None
    return page
