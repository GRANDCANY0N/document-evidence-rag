from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mineru_vlm_rag.chunking.builder import CHUNK_SCHEMA_VERSION
from mineru_vlm_rag.domain.models import BlockType, Chunk, Document, ResolutionStatus


_FAILURE_PLACEHOLDER_RE = re.compile(
    r"^(?:the\s+(?:image|text).*(?:too\s+blurry|cannot|can't|unable).*(?:recognize|read)|"
    r"(?:图片|图像|文字).*(?:太模糊|无法|不能).*(?:识别|读取)|"
    r"(?:无法|不能)(?:识别|读取).*(?:图片|图像|文字))",
    re.I | re.S,
)


def build_chunk_export(document: Document, chunks: list[Chunk], *, parameters: dict[str, Any]) -> dict[str, Any]:
    role_counts = Counter(str(chunk.metadata.get("chunk_role") or "unknown") for chunk in chunks)
    modality_counts = Counter(chunk.block_type for chunk in chunks)
    parent_count = len({
        str(chunk.metadata.get("logical_parent_id"))
        for chunk in chunks
        if chunk.metadata.get("logical_parent_id")
    })
    source_blocks = [block for page in document.pages for block in page.blocks]
    status_counts = Counter(block.status.value for block in source_blocks)
    source_type_counts = Counter(block.block_type.value for block in source_blocks)
    excluded_reason_counts: Counter[str] = Counter()
    represented_block_ids = {
        str(block_id)
        for chunk in chunks
        for key in ("source_block_ids", "section_block_ids")
        for block_id in (chunk.metadata.get(key) or [])
    }
    excluded_blocks: list[dict[str, Any]] = []
    unrepresented_eligible: list[str] = []
    provisional_represented = 0
    for block in source_blocks:
        represented = block.block_id in represented_block_ids
        if represented and (
            block.status.value not in {"accepted", "repaired"}
            or block.metadata.get("chunk_eligible") is False
        ):
            provisional_represented += 1
        reasons: list[str] = []
        if block.metadata.get("skip_chunk"):
            reasons.append("skip_chunk")
        if block.metadata.get("chunk_eligible") is False:
            reasons.append("chunk_eligible_false")
        if block.status.value not in {"accepted", "repaired"}:
            reasons.append(f"status_{block.status.value}")
        if block.block_type.value in {"header", "footer", "page_number"}:
            reasons.append(f"template_{block.block_type.value}")
        if not block.content.strip():
            reasons.append("empty_content")
        compact_content = re.sub(r"\s+", " ", block.content).strip()
        if len(compact_content) <= 240 and _FAILURE_PLACEHOLDER_RE.search(compact_content):
            reasons.append("model_failure_placeholder")
        intentionally_context_only = (
            block.block_type.value == "title"
            or bool(block.metadata.get("associated_table_block_id"))
        )
        if not reasons and not represented and not intentionally_context_only:
            reasons.append("eligible_semantic_block_not_emitted")
            unrepresented_eligible.append(block.block_id)
        if represented:
            reasons = []
        for reason in reasons:
            excluded_reason_counts[reason] += 1
        if reasons:
            excluded_blocks.append({
                "block_id": block.block_id,
                "page_no": block.page_no,
                "block_type": block.block_type.value,
                "status": block.status.value,
                "source": block.source,
                "reasons": reasons,
                "issue_flags": block.issue_flags,
                "bbox": block.bbox.as_list() if block.bbox else None,
                "content": block.content,
            })
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    valid_statuses = {"accepted", "repaired"}
    validation = {
        "chunk_ids_unique": len(chunk_ids) == len(set(chunk_ids)),
        "all_chunks_nonempty": all(chunk.display_text.strip() and chunk.embedding_text.strip() for chunk in chunks),
        "all_chunks_have_source_blocks": all(chunk.metadata.get("source_block_ids") for chunk in chunks),
        "all_chunk_statuses_quality_tiered": all(
            (
                set(str(value) for value in (chunk.metadata.get("resolution_statuses") or [])) <= valid_statuses
                and chunk.metadata.get("retrieval_tier") == "verified"
            )
            or (
                chunk.metadata.get("retrieval_tier") == "provisional"
                and chunk.metadata.get("requires_human_review") is True
            )
            for chunk in chunks
        ),
        "all_eligible_semantic_blocks_represented": not unrepresented_eligible,
        "embedding_not_executed": True,
    }
    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_executed": False,
        "database_chunks_replaced": False,
        "document": {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "source_path": document.source_path,
            "page_count": document.page_count,
            "status": document.status,
            "mineru_batch_id": document.mineru_batch_id,
        },
        "parameters": parameters,
        "summary": {
            "chunk_count": len(chunks),
            "logical_parent_count": parent_count,
            "role_counts": dict(sorted(role_counts.items())),
            "modality_counts": dict(sorted(modality_counts.items())),
            "source_block_count": len(source_blocks),
            "source_block_status_counts": dict(sorted(status_counts.items())),
            "source_block_type_counts": dict(sorted(source_type_counts.items())),
            "excluded_block_reason_counts": dict(sorted(excluded_reason_counts.items())),
            "provisional_represented_block_count": provisional_represented,
        },
        "validation": validation,
        "excluded_blocks": excluded_blocks,
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }


def write_chunk_export(
    document: Document,
    chunks: list[Chunk],
    output_path: Path,
    *,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    payload = build_chunk_export(document, chunks, parameters=parameters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def build_linear_export(document: Document) -> dict[str, Any]:
    """Export one non-duplicated unit per source block in article order."""
    units: list[dict[str, Any]] = []
    for page in sorted(document.pages, key=lambda value: value.page_no):
        for block in sorted(page.blocks, key=lambda value: value.reading_order):
            if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
                continue
            content_basis = "resolved"
            content = block.content.strip()
            if (
                block.status not in {ResolutionStatus.ACCEPTED, ResolutionStatus.REPAIRED}
                or block.metadata.get("chunk_eligible") is False
            ) and block.raw_content.strip():
                content = block.raw_content.strip()
                content_basis = "mineru_raw"
            if not content:
                continue
            compact_content = re.sub(r"\s+", " ", content).strip()
            if len(compact_content) <= 240 and _FAILURE_PLACEHOLDER_RE.search(compact_content):
                continue
            units.append({
                "article_order": len(units),
                "page_no": page.page_no,
                "reading_order": block.reading_order,
                "mineru_order": block.metadata.get("mineru_order"),
                "block_id": block.block_id,
                "block_type": block.block_type.value,
                "status": block.status.value,
                "retrieval_tier": (
                    "verified"
                    if block.status in {ResolutionStatus.ACCEPTED, ResolutionStatus.REPAIRED}
                    and block.metadata.get("chunk_eligible") is not False
                    else "provisional"
                ),
                "content_basis": content_basis,
                "section_path": block.section_path,
                "bbox": block.bbox.as_list() if block.bbox else None,
                "associated_table_block_id": block.metadata.get("associated_table_block_id"),
                "skip_chunk": bool(block.metadata.get("skip_chunk")),
                "issue_flags": block.issue_flags,
                "content": content,
            })
    return {
        "schema_version": "linear-document-units-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_id": document.document_id,
        "description": "One source block per item, no table-row/chart-fact fan-out; use this file to inspect article order.",
        "unit_count": len(units),
        "units": units,
    }


def write_linear_export(document: Document, output_path: Path) -> dict[str, Any]:
    payload = build_linear_export(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload
