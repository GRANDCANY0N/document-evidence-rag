from __future__ import annotations

import re

from mineru_vlm_rag.domain.models import Block, BlockType, Document


def _normalized_template_text(text: str) -> str:
    value = re.sub(r"\s+", "", text).strip().lower()
    value = re.sub(r"\d+", "#", value)
    return value[:160]


def _coordinate_height(block: Block, page_height: float | None, page_max_y: float) -> float:
    if page_height and page_max_y <= page_height * 1.5:
        return page_height
    return page_max_y or page_height or 1.0


def mark_repeated_templates(document: Document) -> None:
    """Exclude repeated headers/footers/page numbers from semantic chunks without deleting evidence."""
    if len(document.pages) < 2:
        return
    per_page_candidates: list[tuple[Block, str, float]] = []
    page_sets: dict[str, set[int]] = {}
    for page in document.pages:
        positioned = [block for block in page.blocks if block.bbox and block.content.strip()]
        page_max_y = max((block.bbox.y1 for block in positioned if block.bbox), default=0.0)
        for block in positioned:
            key = _normalized_template_text(block.content)
            if not key or len(key) > 120:
                continue
            height = _coordinate_height(block, page.height, page_max_y)
            y_ratio = ((block.bbox.y0 + block.bbox.y1) / 2) / height
            per_page_candidates.append((block, key, y_ratio))
            page_sets.setdefault(key, set()).add(page.page_no)

    minimum_pages = max(2, int(len(document.pages) * 0.6 + 0.5))
    repeated = {key for key, pages in page_sets.items() if len(pages) >= minimum_pages}
    for block, key, y_ratio in per_page_candidates:
        compact = re.sub(r"\s+", "", block.content)
        is_page_number = bool(re.fullmatch(r"(?:第)?[-—–]?\s*\d+\s*(?:页|/\s*\d+)?", compact))
        if is_page_number and y_ratio >= 0.78:
            block.block_type = BlockType.PAGE_NUMBER
            block.metadata["template_kind"] = "page_number"
        elif key in repeated and y_ratio <= 0.14:
            block.block_type = BlockType.HEADER
            block.metadata["template_kind"] = "repeated_header"
        elif key in repeated and y_ratio >= 0.86:
            block.block_type = BlockType.FOOTER
            block.metadata["template_kind"] = "repeated_footer"
        else:
            continue
        block.metadata["skip_chunk"] = True
        block.issue_flags = list(dict.fromkeys([*block.issue_flags, "LAY-003"]))


def mark_cross_page_duplicate_fragments(document: Document) -> int:
    """Suppress exact page-top carry-over text while preserving the source block.

    Some PDF extractors include the end of page N both in the final block of
    page N and in a small block at the top of page N+1.  This is not missing
    content and must not become a second embedding.  We only suppress exact,
    whitespace-insensitive matches near the page top; fuzzy matches remain
    untouched because they may contain new evidence.
    """
    pages = sorted(document.pages, key=lambda value: value.page_no)
    suppressed = 0
    for previous_page, current_page in zip(pages, pages[1:]):
        if current_page.page_no != previous_page.page_no + 1:
            continue
        previous_blocks = [
            block for block in previous_page.blocks
            if block.content.strip()
            and block.block_type not in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
        ]
        previous_compact = [(_compact(block.content), block) for block in previous_blocks]
        page_height = _coordinate_size(current_page, "height")
        candidates = sorted(
            [
                block for block in current_page.blocks
                if block.block_type == BlockType.TEXT
                and block.content.strip()
                and block.bbox
                and not block.metadata.get("associated_table_block_id")
                and block.bbox.y0 <= page_height * 0.20
            ],
            key=lambda block: (block.bbox.y0, block.bbox.x0),
        )[:3]
        for block in candidates:
            compact = _compact(block.content)
            if len(compact) < 5 or len(compact) > 220:
                continue
            duplicate_of = next(
                (
                    previous
                    for previous_text, previous in reversed(previous_compact)
                    if compact in previous_text
                    and (
                        len(compact) >= 12
                        or previous_text.endswith(compact)
                    )
                ),
                None,
            )
            if duplicate_of is None:
                continue
            block.metadata["skip_chunk"] = True
            block.metadata["cross_page_duplicate_of"] = duplicate_of.block_id
            block.metadata["cross_page_duplicate_from_page"] = previous_page.page_no
            tags = [str(value) for value in (block.metadata.get("processing_tags") or [])]
            block.metadata["processing_tags"] = list(dict.fromkeys([
                *tags,
                "cross_page_duplicate_suppressed",
            ]))
            suppressed += 1
    return suppressed


def _title_level(text: str) -> int:
    compact = text.strip()
    if re.match(r"^第[一二三四五六七八九十百\d]+[编章节篇]", compact):
        return 1
    if re.match(r"^(?:专栏|专题)[一二三四五六七八九十百\d]+", compact):
        return 1
    if re.match(r"^[一二三四五六七八九十]+[、.]", compact):
        return 2
    numeric = re.match(r"^(\d+(?:\.\d+){0,4})[\s、.]", compact)
    if numeric:
        return min(5, numeric.group(1).count(".") + 1)
    if re.match(r"^[（(][一二三四五六七八九十\d]+[)）]", compact):
        return 3
    return 1


_UNIT_RE = re.compile(r"^[（(]?\s*(?:单位|币种)[：:]", re.I)
_DATE_RE = re.compile(r"^[（(]?\s*(?:截至)?\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?[）)]?$")
_NOTE_RE = re.compile(r"^[（(]?\s*(?:注|说明|资料来源|数据来源)[：:]", re.I)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "").strip()


def _context_kind(block: Block) -> str:
    """Classify local visual context without turning it into a chapter."""
    value = block.content.strip()
    compact = _compact(value).strip("（）()")
    if not value:
        return "empty"
    if _NOTE_RE.match(value) or re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", compact) or block.block_type == BlockType.FOOTNOTE:
        return "note"
    if _UNIT_RE.match(value):
        return "unit"
    if _DATE_RE.fullmatch(compact) or re.fullmatch(r"[（(].*(?:余额|日期).*[）)]", value):
        return "date"
    if re.fullmatch(r"(?:续)?(?:附表|表|图)\s*[一二三四五六七八九十百\d]+(?:[-—.]\d+)?", compact, re.I):
        return "reference"
    if re.fullmatch(r"续表", compact, re.I):
        return "continuation"
    return "title"


def _coordinate_size(document_page, axis: str) -> float:
    key = f"mineru_coordinate_{axis}"
    explicit = document_page.metadata.get(key)
    if explicit:
        return float(explicit)
    value = getattr(document_page, axis, None)
    return float(value or 1000.0)


def _vertical_overlap(left: Block, right: Block) -> float:
    if not left.bbox or not right.bbox:
        return 0.0
    return max(0.0, min(left.bbox.y1, right.bbox.y1) - max(left.bbox.y0, right.bbox.y0))


def _is_likely_table_context(block: Block, table: Block, page_height: float, page_width: float) -> bool:
    if not block.bbox or not table.bbox or not block.content.strip():
        return False
    if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
        return False
    existing = str(block.metadata.get("associated_table_block_id") or "")
    if existing and existing != table.block_id:
        return False
    kind = _context_kind(block)
    above_gap = table.bbox.y0 - block.bbox.y1
    horizontally_aligned = not (block.bbox.x1 < table.bbox.x0 or block.bbox.x0 > table.bbox.x1)
    if 0 <= above_gap <= page_height * 0.09 and horizontally_aligned:
        compact = _compact(block.content)
        caption_like = bool(
            len(compact) <= 50
            and "\n" not in block.content
            and not re.search(r"[。！？；]", block.content)
        )
        return bool(
            kind != "title"
            or block.block_type == BlockType.TITLE
            or block.metadata.get("text_level") is not None
            or caption_like
        )
    below_gap = block.bbox.y0 - table.bbox.y1
    if kind == "note" and 0 <= below_gap <= page_height * 0.08 and horizontally_aligned:
        return True

    # Rotated statistical appendices put title/unit/note alongside a tall
    # table.  Bind only close, vertically overlapping context so ordinary
    # two-column prose is not stolen by the table.
    if block.bbox.height >= block.bbox.width * 1.4:
        side_gap = min(abs(table.bbox.x0 - block.bbox.x1), abs(block.bbox.x0 - table.bbox.x1))
        overlap = _vertical_overlap(block, table)
        if side_gap <= page_width * 0.08 and overlap >= min(block.bbox.height, table.bbox.height) * 0.25:
            return kind != "title" or block.bbox.height >= block.bbox.width * 1.4
    return False


def _caption_score(block: Block, table: Block) -> tuple[int, float, int]:
    kind = _context_kind(block)
    kind_rank = {"title": 0, "reference": 1, "date": 2, "unit": 3, "note": 4}.get(kind, 5)
    if not block.bbox or not table.bbox:
        gap = float("inf")
    elif block.bbox.y1 <= table.bbox.y0:
        gap = table.bbox.y0 - block.bbox.y1
    else:
        gap = abs(table.bbox.x0 - block.bbox.x1)
    # Prefer a descriptive title over a bare "附表 7" identifier.
    descriptive = 0 if kind == "title" and len(_compact(block.content)) >= 4 else 1
    return descriptive, gap, kind_rank


def assign_section_paths(document: Document) -> None:
    """Propagate true document headings; never promote table context to a section."""
    stack: list[str] = []
    previous_heading: Block | None = None
    for page in document.pages:
        for block in page.blocks:
            block.section_path = []
            block.metadata.pop("is_section_heading", None)
            block.metadata.pop("title_level", None)

    for page in sorted(document.pages, key=lambda value: value.page_no):
        for block in sorted(page.blocks, key=lambda value: value.reading_order):
            content = block.content.strip()
            is_heading_candidate = bool(
                content
                and not block.metadata.get("associated_table_block_id")
                and (
                    block.block_type == BlockType.TITLE
                    or block.metadata.get("text_level") is not None
                )
                and _context_kind(block) == "title"
                and len(_compact(content)) <= 100
            )
            if is_heading_candidate:
                level = _title_level(block.content)
                # MinerU sometimes returns "第二章" and its name as two
                # consecutive heading blocks.  Join the section label instead
                # of replacing it with the plain name.
                if (
                    previous_heading is not None
                    and previous_heading.page_no == block.page_no
                    and re.fullmatch(r"第[一二三四五六七八九十百\d]+[编章节篇]", _compact(previous_heading.content))
                    and level == 1
                    and not re.match(r"^(?:第|专栏|专题)", _compact(content))
                ):
                    combined = f"{previous_heading.content.strip()} {content}"
                    stack = [combined]
                    previous_heading.section_path = list(stack)
                    previous_heading.metadata["combined_heading_block_id"] = block.block_id
                    block.metadata["combined_with_heading_block_id"] = previous_heading.block_id
                    block.metadata["is_section_heading"] = True
                    block.metadata["title_level"] = 1
                    block.section_path = list(stack)
                    previous_heading = block
                    continue
                stack = stack[: level - 1]
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(content)
                block.metadata["title_level"] = level
                block.metadata["is_section_heading"] = True
                block.section_path = [item for item in stack if item]
                previous_heading = block
            else:
                block.section_path = [item for item in stack if item]
                if content and block.block_type not in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
                    previous_heading = None


def associate_visual_context(document: Document) -> None:
    """Attach table/title/unit/note evidence without making it a section heading."""
    visual_types = {BlockType.IMAGE, BlockType.CHART, BlockType.TABLE, BlockType.FORMULA}
    context_types = {BlockType.TITLE, BlockType.TEXT, BlockType.FOOTNOTE, BlockType.HEADER}
    for page in document.pages:
        candidates = [
            block for block in page.blocks
            if block.block_type in context_types
            and block.content.strip()
            and block.bbox
            and not block.metadata.get("skip_chunk")
            and block.metadata.get("template_kind") != "page_number"
        ]
        for visual in [block for block in page.blocks if block.block_type in visual_types and block.bbox]:
            page_height = _coordinate_size(page, "height")
            page_width = _coordinate_size(page, "width")
            if visual.block_type == BlockType.TABLE:
                related = [
                    block for block in candidates
                    if block.block_id != visual.block_id
                    and _is_likely_table_context(block, visual, page_height, page_width)
                ]
                for context in related:
                    context.metadata["associated_table_block_id"] = visual.block_id
                    context.metadata["table_context_kind"] = _context_kind(context)
                if related:
                    visual.metadata["context_block_ids"] = [block.block_id for block in related]
                    caption_candidates = [
                        block for block in related
                        if _context_kind(block) in {"title", "reference"}
                    ]
                    if caption_candidates:
                        best = min(caption_candidates, key=lambda block: _caption_score(block, visual))
                        visual.metadata["caption"] = best.content.strip()
                continue

            preceding = [
                block for block in candidates
                if block.bbox and block.bbox.y1 <= visual.bbox.y0
            ]
            if preceding:
                nearest = max(preceding, key=lambda block: block.bbox.y1)
                if visual.bbox.y0 - nearest.bbox.y1 <= page_height * 0.12:
                    visual.metadata["caption"] = nearest.content.strip()
                    visual.metadata["context_block_ids"] = [nearest.block_id]


def normalize_document_structure(document: Document) -> dict[str, int]:
    """Recompute associations, reading order and section paths in a safe order."""
    # Local import avoids a cycle while keeping the public normalization API
    # in one place.
    from mineru_vlm_rag.normalization.reading_order import rebuild_reading_order

    associate_visual_context(document)
    low_confidence_pages = 0
    for page in document.pages:
        rebuild_reading_order(page)
        if page.metadata.get("reading_order_confidence") == "low":
            low_confidence_pages += 1
    duplicate_fragment_count = mark_cross_page_duplicate_fragments(document)
    assign_section_paths(document)
    return {
        "page_count": len(document.pages),
        "low_confidence_page_count": low_confidence_pages,
        "section_heading_count": sum(
            1
            for page in document.pages
            for block in page.blocks
            if block.metadata.get("is_section_heading")
        ),
        "table_context_block_count": sum(
            1
            for page in document.pages
            for block in page.blocks
            if block.metadata.get("associated_table_block_id")
        ),
        "cross_page_duplicate_fragment_count": duplicate_fragment_count,
    }
