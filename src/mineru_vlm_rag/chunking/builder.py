from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from lxml import html

from mineru_vlm_rag.domain.models import Block, BlockType, Chunk, Document, ResolutionStatus
from mineru_vlm_rag.tables import table_records


CHUNK_SCHEMA_VERSION = "multimodal-chunks-v3"
ELIGIBLE_STATUSES = {ResolutionStatus.ACCEPTED, ResolutionStatus.REPAIRED}
TEMPLATE_TYPES = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
_TERMINAL_RE = re.compile(r"[。！？!?；;][\"”’」』）)]?\s*$")
_FACT_RE = re.compile(
    r"(?:[-+]?\d+(?:[.,]\d+)*(?:%|％)?|同比|环比|上升|下降|增长|减少|最高|最低|"
    r"占比|趋势|峰值|谷值|增加|降低)"
)
_FAILURE_PLACEHOLDER_RE = re.compile(
    r"^(?:the\s+(?:image|text).*(?:too\s+blurry|cannot|can't|unable).*(?:recognize|read)|"
    r"(?:图片|图像|文字).*(?:太模糊|无法|不能).*(?:识别|读取)|"
    r"(?:无法|不能)(?:识别|读取).*(?:图片|图像|文字))",
    re.I | re.S,
)


@dataclass(frozen=True)
class _TextAtom:
    text: str
    block: Block


def _stable_id(document_id: str, kind: str, *parts: str) -> str:
    payload = ":".join((document_id, kind, *parts))
    return sha256(payload.encode("utf-8")).hexdigest()[:32]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_failure_placeholder(value: str) -> bool:
    compact = re.sub(r"\s+", " ", value or "").strip()
    return bool(compact and len(compact) <= 240 and _FAILURE_PLACEHOLDER_RE.search(compact))


def _is_provisional(block: Block) -> bool:
    return block.status not in ELIGIBLE_STATUSES or block.metadata.get("chunk_eligible") is False


def _chunk_content(block: Block) -> str:
    """Use exact MinerU evidence for unresolved blocks, never a guessed revision."""
    if _is_provisional(block) and block.raw_content.strip():
        return block.raw_content
    return block.content


def _source_pages(blocks: list[Block]) -> list[int]:
    pages: list[int] = []
    for block in blocks:
        pages.extend(int(value) for value in (block.metadata.get("source_pages") or [block.page_no]))
    return sorted(set(pages))


def _source_bboxes(blocks: list[Block]) -> list[dict[str, object]]:
    return [
        {"block_id": block.block_id, "page_no": block.page_no, "bbox": block.bbox.as_list()}
        for block in blocks
        if block.bbox
    ]


def _logical_parent_id(document_id: str, role: str, blocks: list[Block]) -> str:
    return _stable_id(document_id, f"logical-{role}", *[block.block_id for block in blocks])


def _make_chunk(
    blocks: list[Block],
    *,
    role: str,
    index: int,
    display_text: str,
    parent_id: str,
    block_type: str | None = None,
    section_path: list[str] | None = None,
    asset_id: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Chunk:
    if not blocks:
        raise ValueError("A chunk must reference at least one source block")
    display_text = display_text.strip()
    if not display_text:
        raise ValueError("A chunk must contain display text")
    document_id = blocks[0].document_id
    section_values = section_path if section_path is not None else blocks[0].section_path
    section = " / ".join(value for value in section_values if value)
    retrieval_tier = "provisional" if any(_is_provisional(block) for block in blocks) else "verified"
    prefix_parts = []
    if retrieval_tier == "provisional":
        prefix_parts.append("证据状态：待复核（保留MinerU原始内容）")
    if section:
        prefix_parts.append(f"章节：{section}")
    embedding_text = "\n".join([*prefix_parts, display_text])
    pages = _source_pages(blocks)
    source_bboxes = _source_bboxes(blocks)
    sources = _dedupe(block.source for block in blocks)
    statuses = _dedupe(block.status.value for block in blocks)
    issue_flags = _dedupe(flag for block in blocks for flag in block.issue_flags)
    processing_tags = _dedupe(
        str(tag) for block in blocks for tag in (block.metadata.get("processing_tags") or [])
    )
    verification_tags = _dedupe(
        str(tag) for block in blocks for tag in (block.metadata.get("verification_tags") or [])
    )
    metadata: dict[str, object] = {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_role": role,
        "logical_parent_id": parent_id,
        "section_path": list(section_values),
        "source_block_ids": [block.block_id for block in blocks],
        "source_pages": pages,
        "source_bboxes": source_bboxes,
        "sources": sources,
        "resolution_statuses": statuses,
        "issue_flags": issue_flags,
        "processing_tags": processing_tags,
        "verification_tags": verification_tags,
        "retrieval_tier": retrieval_tier,
        "requires_human_review": retrieval_tier == "provisional",
        "source_content_basis": {
            block.block_id: "mineru_raw" if _is_provisional(block) and block.raw_content.strip() else "resolved"
            for block in blocks
        },
        "anchor_page": blocks[0].page_no,
        "anchor_reading_order": blocks[0].reading_order,
        "anchor_mineru_order": blocks[0].metadata.get("mineru_order"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    chunk_id = _stable_id(
        document_id,
        role,
        parent_id,
        str(index),
        ",".join(block.block_id for block in blocks),
        display_text,
    )
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_start=min(pages),
        page_end=max(pages),
        block_type=block_type or blocks[0].block_type.value,
        parent_id=parent_id,
        section_path=section,
        display_text=display_text,
        embedding_text=embedding_text,
        asset_id=asset_id if asset_id is not None else next((block.asset_id for block in blocks if block.asset_id), None),
        bbox_json=json.dumps(source_bboxes, ensure_ascii=False, separators=(",", ":")),
        metadata=metadata,
    )


def _split_long_unit(value: str, size: int, overlap: int) -> list[str]:
    value = value.strip()
    if len(value) <= size:
        return [value] if value else []
    clauses = [part.strip() for part in re.split(r"(?<=[，、,:：])", value) if part.strip()]
    if len(clauses) > 1 and max(map(len, clauses)) <= size:
        output: list[str] = []
        current = ""
        for clause in clauses:
            if current and len(current) + len(clause) > size:
                output.append(current)
                current = clause
            else:
                current += clause
        if current:
            output.append(current)
        return output
    step = max(1, size - overlap)
    return [value[offset:offset + size] for offset in range(0, len(value), step)]


def _sentence_units(text: str, size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]:
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", paragraph) if part.strip()]
        for sentence in sentences or [paragraph]:
            units.extend(_split_long_unit(sentence, size, overlap))
    return units


def _render_atoms(atoms: list[_TextAtom]) -> str:
    parts: list[str] = []
    previous_id = ""
    for atom in atoms:
        if parts and atom.block.block_id != previous_id:
            parts.append("\n\n")
        parts.append(atom.text)
        previous_id = atom.block.block_id
    return "".join(parts).strip()


def _atom_windows(blocks: list[Block], size: int, overlap: int) -> list[tuple[str, list[Block]]]:
    atoms = [
        _TextAtom(text=unit, block=block)
        for block in blocks
        for unit in _sentence_units(_chunk_content(block), size, overlap)
    ]
    windows: list[tuple[str, list[Block]]] = []
    current: list[_TextAtom] = []

    def emit() -> None:
        if not current:
            return
        source_ids = list(dict.fromkeys(atom.block.block_id for atom in current))
        lookup = {block.block_id: block for block in blocks}
        windows.append((_render_atoms(current), [lookup[block_id] for block_id in source_ids]))

    for atom in atoms:
        candidate = _render_atoms([*current, atom])
        if current and len(candidate) > size:
            emit()
            tail: list[_TextAtom] = []
            tail_chars = 0
            for previous in reversed(current):
                if tail and tail_chars >= overlap:
                    break
                tail.insert(0, previous)
                tail_chars += len(previous.text)
            current = tail
            while current and len(_render_atoms([*current, atom])) > size:
                current.pop(0)
        current.append(atom)
    emit()
    return windows


def _can_merge_text(previous: Block, current: Block, buffered_chars: int, max_group_chars: int) -> bool:
    if buffered_chars + len(_chunk_content(current)) > max_group_chars:
        return False
    if previous.section_path != current.section_path or previous.status != current.status:
        return False
    if current.page_no == previous.page_no:
        left_column = previous.metadata.get("column_index")
        right_column = current.metadata.get("column_index")
        return left_column is None or right_column is None or left_column == right_column
    if current.page_no != previous.page_no + 1:
        return False
    return not _TERMINAL_RE.search(_chunk_content(previous))


def _text_group_chunks(blocks: list[Block], size: int, overlap: int) -> list[Chunk]:
    if not blocks:
        return []
    parent_id = _logical_parent_id(blocks[0].document_id, "text", blocks)
    chunks: list[Chunk] = []
    for index, (text, source_blocks) in enumerate(_atom_windows(blocks, size, overlap)):
        chunks.append(
            _make_chunk(
                source_blocks,
                role="text_segment",
                index=index,
                display_text=text,
                parent_id=parent_id,
                block_type=BlockType.TEXT.value,
                section_path=blocks[0].section_path,
                extra_metadata={"segment_index": index},
            )
        )
    return chunks


def _flatten_table(content: str) -> str:
    try:
        root = html.fromstring(content)
        return " ".join(" ".join(root.itertext()).split())
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()


def _table_context(block: Block, associated: list[Block]) -> list[str]:
    lines: list[str] = []
    generic_caption = str(block.metadata.get("caption") or "").strip()

    def vertical_gap(context: Block) -> float:
        if not block.bbox or not context.bbox:
            return float("inf")
        if context.bbox.y1 <= block.bbox.y0:
            return block.bbox.y0 - context.bbox.y1
        if context.bbox.y0 >= block.bbox.y1:
            return context.bbox.y0 - block.bbox.y1
        return 0.0

    notes: list[Block] = []
    units: list[Block] = []
    dates: list[Block] = []
    title_candidates: list[Block] = []
    for context in associated:
        value = _chunk_content(context).strip()
        compact = value.strip("（）() ")
        if not value:
            continue
        context_kind = str(context.metadata.get("table_context_kind") or "")
        if context_kind == "note" or context.block_type == BlockType.FOOTNOTE or re.match(
            r"\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（(]?(?:注|说明|资料来源|数据来源)[：:])", value
        ):
            notes.append(context)
        elif re.match(r"\s*[（(]?(?:单位|币种)[：:]", value):
            units.append(context)
        elif re.fullmatch(r"(?:截至)?\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?", compact):
            dates.append(context)
        elif (
            context.block_type == BlockType.TITLE
            or context.metadata.get("text_level") is not None
            or re.match(r"^(?:附表|表|图)\s*\d+", compact)
            or (len(re.sub(r"\s+", "", value)) <= 50 and not re.search(r"[。！？；]", value))
        ):
            title_candidates.append(context)
        else:
            # Long rotated side text is normally a footnote continuation, not
            # part of the table title (pages 113/116 in the reference PDF).
            notes.append(context)

    selected_titles: list[Block] = []
    if title_candidates:
        closest_gap = min(vertical_gap(context) for context in title_candidates)
        selected_titles = [
            context
            for context in title_candidates
            if vertical_gap(context) <= max(30.0, closest_gap + 20.0)
        ]
        selected_titles.sort(key=lambda context: (
            context.bbox.y0 if context.bbox else float("inf"),
            context.reading_order,
        ))
    if selected_titles:
        lines.append("表题：" + " ".join(_chunk_content(context).strip() for context in selected_titles))
    elif generic_caption:
        lines.append(f"表题：{generic_caption}")

    for context in dates:
        if vertical_gap(context) <= 35.0:
            lines.append(f"日期：{_chunk_content(context).strip()}")
    for context in units:
        if vertical_gap(context) <= 45.0:
            value = re.sub(
                r"^\s*[（(]?(?:单位|币种)[：:]\s*",
                "",
                _chunk_content(context).strip(),
            ).rstrip("）) ")
            lines.append(f"单位：{value}")
    for context in notes:
        # A source note below a table belongs to that table. An above-table
        # note is accepted only when it is immediately adjacent; otherwise it
        # commonly belongs to the preceding chart/table.
        below_table = bool(block.bbox and context.bbox and context.bbox.y0 >= block.bbox.y1)
        if below_table or vertical_gap(context) <= 30.0:
            lines.append(f"表注：{_chunk_content(context).strip()}")
    return _dedupe(lines)


def _record_text(record: dict[str, object]) -> str:
    cells = record.get("cells") or {}
    assert isinstance(cells, dict)
    parts = []
    if record.get("source_page"):
        parts.append(f"来源页：{record['source_page']}")
    parts.append("数据行：" + "；".join(f"{key}={value}" for key, value in cells.items() if value))
    return "\n".join(parts)


def _table_chunks(
    block: Block,
    associated: list[Block],
    *,
    size: int,
    row_group_size: int,
) -> list[Chunk]:
    evidence_blocks = [block, *associated]
    logical_id = _logical_parent_id(block.document_id, "table", evidence_blocks)
    context_lines = _table_context(block, associated)
    context = "\n".join(context_lines)
    try:
        headers, records = table_records(_chunk_content(block))
    except Exception:
        headers, records = [], []
    if not records:
        fallback = _flatten_table(_chunk_content(block))
        texts = _sentence_units(fallback, size, max(1, size // 10))
        return [
            _make_chunk(
                evidence_blocks,
                role="table_fallback",
                index=index,
                display_text="\n".join(part for part in (context, text) if part),
                parent_id=logical_id,
                block_type=BlockType.TABLE.value,
                extra_metadata={"table_parseable": False, "segment_index": index},
            )
            for index, text in enumerate(texts)
            if text
        ]

    first_header = headers[0] if headers else ""
    row_labels = _dedupe(
        str((record.get("cells") or {}).get(first_header) or "")
        for record in records
        if first_header
    )
    overview_parts = [
        context,
        "表格概览",
        f"表头路径：{' | '.join(headers)}",
        f"数据行数：{len(records)}",
    ]
    if row_labels:
        overview_parts.append("主要项目：" + "；".join(row_labels[:40]))
    overview = _make_chunk(
        evidence_blocks,
        role="table_summary",
        index=0,
        display_text="\n".join(part for part in overview_parts if part),
        parent_id=logical_id,
        block_type=BlockType.TABLE.value,
        extra_metadata={
            "table_parseable": True,
            "header_paths": headers,
            "row_count": len(records),
            "is_parent_summary": True,
        },
    )
    chunks = [overview]
    child_parent = overview.chunk_id

    if len(records) > 1:
        groups: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        for record in records:
            candidate = [*current, record]
            candidate_text = "\n".join(_record_text(item) for item in candidate)
            if current and (len(current) >= row_group_size or len(candidate_text) > size):
                groups.append(current)
                current = [record]
            else:
                current = candidate
        if current:
            groups.append(current)
        for index, group in enumerate(groups):
            # A final singleton group is byte-for-byte identical to the exact
            # row chunk emitted below. Do not create a duplicate vector.
            if len(group) < 2:
                continue
            row_indexes = [int(record.get("row_index") or 0) for record in group]
            source_pages = sorted({int(record["source_page"]) for record in group if record.get("source_page")})
            body = "\n".join(_record_text(record) for record in group)
            chunks.append(
                _make_chunk(
                    evidence_blocks,
                    role="table_row_group",
                    index=index,
                    display_text="\n".join(part for part in (context, f"表头路径：{' | '.join(headers)}", body) if part),
                    parent_id=child_parent,
                    block_type=BlockType.TABLE.value,
                    extra_metadata={
                        "header_paths": headers,
                        "row_indexes": row_indexes,
                        "record_source_pages": source_pages,
                    },
                )
            )

    for index, record in enumerate(records):
        chunks.append(
            _make_chunk(
                evidence_blocks,
                role="table_row",
                index=index,
                display_text="\n".join(
                    part for part in (context, f"表头路径：{' | '.join(headers)}", _record_text(record)) if part
                ),
                parent_id=child_parent,
                block_type=BlockType.TABLE.value,
                extra_metadata={
                    "header_paths": headers,
                    "row_index": int(record.get("row_index") or 0),
                    "record_source_page": record.get("source_page"),
                    "cells": record.get("cells") or {},
                },
            )
        )
    return chunks


def _visual_chunks(block: Block, *, size: int, overlap: int) -> list[Chunk]:
    prefix = "图表" if block.block_type == BlockType.CHART else "图片"
    caption = str(block.metadata.get("caption") or "").strip()
    caption_label = "图题" if len(caption) <= 80 and caption.count("。") <= 1 else "相关正文"
    context = f"{caption_label}：{caption}" if caption else ""
    fact_context = (
        context
        if caption_label == "图题"
        else (f"相关主题：{caption[:120]}" if caption else "")
    )
    content = _chunk_content(block).strip()
    if not content:
        return []
    logical_id = _logical_parent_id(block.document_id, block.block_type.value, [block])
    overview = _make_chunk(
        [block],
        role=f"{block.block_type.value}_summary",
        index=0,
        display_text="\n".join(part for part in (context, f"{prefix}描述：{content}") if part),
        parent_id=logical_id,
        extra_metadata={"is_parent_summary": True},
    )
    chunks = [overview]
    if block.block_type == BlockType.CHART:
        facts = [value for value in _sentence_units(content, min(size, 500), overlap) if _FACT_RE.search(value)]
        if len(facts) > 1:
            for index, fact in enumerate(facts):
                chunks.append(
                    _make_chunk(
                        [block],
                        role="chart_fact",
                        index=index,
                        display_text="\n".join(part for part in (fact_context, f"图表事实：{fact}") if part),
                        parent_id=overview.chunk_id,
                        extra_metadata={"fact_index": index},
                    )
                )
    elif len(content) > size:
        for index, value in enumerate(_sentence_units(content, size, overlap)):
            chunks.append(
                _make_chunk(
                    [block],
                    role="image_text_segment",
                    index=index,
                    display_text="\n".join(part for part in (context, value) if part),
                    parent_id=overview.chunk_id,
                    extra_metadata={"segment_index": index},
                )
            )
    return chunks


def _formula_chunk(block: Block, previous: Block | None, following: Block | None) -> Chunk | None:
    if not _chunk_content(block).strip():
        return None
    context_blocks = [value for value in (previous, block, following) if value is not None]
    logical_id = _logical_parent_id(block.document_id, "formula", context_blocks)
    parts = []
    if previous and previous.block_type == BlockType.TEXT:
        parts.append("前文：" + _chunk_content(previous).strip()[-240:])
    parts.append("公式：" + _chunk_content(block).strip())
    if following and following.block_type == BlockType.TEXT:
        parts.append("后文：" + _chunk_content(following).strip()[:240])
    return _make_chunk(
        context_blocks,
        role="formula_context",
        index=0,
        display_text="\n".join(parts),
        parent_id=logical_id,
        block_type=BlockType.FORMULA.value,
        asset_id=block.asset_id,
        section_path=block.section_path,
    )


def _standalone_chunk(block: Block, role: str, label: str) -> Chunk | None:
    if not _chunk_content(block).strip():
        return None
    logical_id = _logical_parent_id(block.document_id, role, [block])
    return _make_chunk(
        [block],
        role=role,
        index=0,
        display_text=f"{label}：{_chunk_content(block).strip()}",
        parent_id=logical_id,
    )


def _eligible(block: Block) -> bool:
    if block.metadata.get("skip_chunk") or block.block_type in TEMPLATE_TYPES:
        return False
    content = _chunk_content(block).strip()
    if not content or _is_failure_placeholder(content):
        return False
    if block.status in ELIGIBLE_STATUSES and block.metadata.get("chunk_eligible") is not False:
        return True
    # A non-empty MinerU original is valuable retrieval evidence even when
    # later VLM/PDF checks disagree. It is emitted as a provisional chunk and
    # never silently replaced by the conflicting post-process result.
    return bool(block.raw_content.strip())


def _link_chunks(chunks: list[Chunk], title_ids_by_path: dict[tuple[str, ...], str]) -> None:
    siblings: dict[str, list[Chunk]] = defaultdict(list)
    local_orders: dict[tuple[int, int], int] = defaultdict(int)
    for order, chunk in enumerate(chunks):
        anchor_key = (
            int(chunk.metadata.get("anchor_page") or chunk.page_start),
            int(chunk.metadata.get("anchor_reading_order") or 0),
        )
        local_role_order = local_orders[anchor_key]
        local_orders[anchor_key] += 1
        chunk.metadata["document_order"] = order
        chunk.metadata["retrieval_order"] = order
        chunk.metadata["local_role_order"] = local_role_order
        chunk.metadata["article_order_key"] = [*anchor_key, local_role_order]
        chunk.metadata["previous_chunk_id"] = chunks[order - 1].chunk_id if order else None
        chunk.metadata["next_chunk_id"] = chunks[order + 1].chunk_id if order + 1 < len(chunks) else None
        siblings[chunk.parent_id or ""].append(chunk)
    for values in siblings.values():
        for index, chunk in enumerate(values):
            chunk.metadata["sibling_index"] = index
            chunk.metadata["previous_sibling_id"] = values[index - 1].chunk_id if index else None
            chunk.metadata["next_sibling_id"] = values[index + 1].chunk_id if index + 1 < len(values) else None
    for chunk in chunks:
        path = tuple(str(value) for value in (chunk.metadata.get("section_path") or []))
        chunk.metadata["section_block_ids"] = [
            title_ids_by_path[path[:depth]]
            for depth in range(1, len(path) + 1)
            if path[:depth] in title_ids_by_path
        ]


def build_chunks(
    document: Document,
    text_size: int = 1200,
    overlap: int = 120,
    *,
    table_row_group_size: int = 6,
) -> list[Chunk]:
    """Build evidence-linked multimodal chunks without calling an embedding model."""
    ordered = [
        block
        for page in sorted(document.pages, key=lambda value: value.page_no)
        for block in sorted(page.blocks, key=lambda value: value.reading_order)
    ]
    associated_by_table: dict[str, list[Block]] = defaultdict(list)
    attached_context_ids: set[str] = set()
    title_ids_by_path: dict[tuple[str, ...], str] = {}
    for block in ordered:
        table_id = str(block.metadata.get("associated_table_block_id") or "")
        if table_id:
            associated_by_table[table_id].append(block)
            attached_context_ids.add(block.block_id)
        if block.block_type == BlockType.TITLE and block.section_path and _eligible(block):
            title_ids_by_path[tuple(block.section_path)] = block.block_id

    chunks: list[Chunk] = []
    text_buffer: list[Block] = []
    buffered_chars = 0
    max_group_chars = max(text_size, text_size * 3)

    def flush_text() -> None:
        nonlocal buffered_chars
        if text_buffer:
            chunks.extend(_text_group_chunks(text_buffer, text_size, overlap))
            text_buffer.clear()
            buffered_chars = 0

    for position, block in enumerate(ordered):
        if not _eligible(block):
            if block.block_type not in TEMPLATE_TYPES:
                flush_text()
            continue
        if block.block_id in attached_context_ids:
            flush_text()
            continue
        if block.block_type == BlockType.TITLE:
            flush_text()
            continue
        if block.block_type == BlockType.TEXT:
            if text_buffer and not _can_merge_text(text_buffer[-1], block, buffered_chars, max_group_chars):
                flush_text()
            text_buffer.append(block)
            buffered_chars += len(_chunk_content(block))
            continue

        flush_text()
        if block.block_type == BlockType.TABLE:
            chunks.extend(
                _table_chunks(
                    block,
                    associated_by_table.get(block.block_id, []),
                    size=text_size,
                    row_group_size=max(1, table_row_group_size),
                )
            )
        elif block.block_type in {BlockType.IMAGE, BlockType.CHART}:
            chunks.extend(_visual_chunks(block, size=text_size, overlap=overlap))
        elif block.block_type == BlockType.FORMULA:
            previous = next((
                value for value in reversed(ordered[:position])
                if _eligible(value)
                and value.block_type == BlockType.TEXT
                and value.page_no == block.page_no
                and value.section_path == block.section_path
            ), None)
            following = next((
                value for value in ordered[position + 1:]
                if _eligible(value)
                and value.block_type == BlockType.TEXT
                and value.page_no == block.page_no
                and value.section_path == block.section_path
            ), None)
            chunk = _formula_chunk(block, previous, following)
            if chunk:
                chunks.append(chunk)
        elif block.block_type == BlockType.FOOTNOTE:
            chunk = _standalone_chunk(block, "footnote", "脚注")
            if chunk:
                chunks.append(chunk)
        elif block.block_type == BlockType.SEAL:
            chunk = _standalone_chunk(block, "seal", "印章")
            if chunk:
                chunks.append(chunk)
        elif _chunk_content(block).strip():
            chunk = _standalone_chunk(block, "other_semantic", "内容")
            if chunk:
                chunks.append(chunk)
    flush_text()
    _link_chunks(chunks, title_ids_by_path)
    return chunks
