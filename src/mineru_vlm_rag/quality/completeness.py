from __future__ import annotations

import html as html_std
import json
import logging
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lxml import etree, html
from PIL import Image

from mineru_vlm_rag.domain.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    ResolutionStatus,
    Revision,
)
from mineru_vlm_rag.pdf import render_page


LOGGER = logging.getLogger(__name__)

_TRANSLATION = str.maketrans(
    {
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        "“": '"', "”": '"', "‘": "'", "’": "'", "﹒": ".",
        "\u00ad": "", "\u200b": "", "\ufeff": "",
    }
)
_NON_SEMANTIC_TYPES = {
    BlockType.IMAGE, BlockType.CHART, BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER,
}


def normalize_text(value: str) -> str:
    value = html_std.unescape(value or "")
    value = unicodedata.normalize("NFKC", value).translate(_TRANSLATION)
    return re.sub(r"\s+", "", value)


def flatten_content(value: str) -> str:
    if not value:
        return ""
    if "<table" not in value.lower():
        return value
    try:
        root = html.fromstring(value)
        return " ".join(" ".join(root.itertext()).split())
    except Exception:
        return re.sub(r"<[^>]+>", " ", value)


def number_tokens(value: str) -> list[str]:
    value = html_std.unescape(value or "")
    value = value.replace("\\%", "%")
    value = unicodedata.normalize("NFKC", value).translate(_TRANSLATION).replace("−", "-")
    return re.findall(r"[-+]?\d+(?:[.,]\d+)*(?:%)?", value)


@dataclass(frozen=True)
class SourceLine:
    page_no: int
    line_no: int
    text: str
    bbox: BoundingBox

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)


@dataclass(frozen=True)
class TextLayer:
    pages: dict[int, list[SourceLine]]
    word_count: int
    invalid_bbox_count: int


def extract_pdf_text_layer(pdf_path: Path) -> TextLayer:
    completed = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf_path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    root = etree.fromstring(
        completed.stdout,
        parser=etree.XMLParser(recover=True, huge_tree=True),
    )
    pages: dict[int, list[SourceLine]] = {}
    word_count = 0
    invalid_bbox_count = 0
    for page_no, page_node in enumerate(root.xpath("//*[local-name()='page']"), start=1):
        width = float(page_node.get("width") or 0)
        height = float(page_node.get("height") or 0)
        page_lines: list[SourceLine] = []
        if width <= 0 or height <= 0:
            pages[page_no] = page_lines
            continue
        for line_no, line_node in enumerate(page_node.xpath(".//*[local-name()='line']")):
            words: list[tuple[str, float, float, float, float]] = []
            for word_node in line_node.xpath("./*[local-name()='word']"):
                text = "".join(word_node.itertext()).strip()
                if not text:
                    continue
                word_count += 1
                try:
                    x0 = float(word_node.get("xMin")) / width * 1000.0
                    y0 = float(word_node.get("yMin")) / height * 1000.0
                    x1 = float(word_node.get("xMax")) / width * 1000.0
                    y1 = float(word_node.get("yMax")) / height * 1000.0
                except (TypeError, ValueError):
                    invalid_bbox_count += 1
                    continue
                if x1 <= x0 or y1 <= y0:
                    invalid_bbox_count += 1
                    continue
                words.append((text, x0, y0, x1, y1))
            if not words:
                continue
            page_lines.append(
                SourceLine(
                    page_no=page_no,
                    line_no=line_no,
                    text=" ".join(word[0] for word in words),
                    bbox=BoundingBox(
                        x0=min(word[1] for word in words),
                        y0=min(word[2] for word in words),
                        x1=max(word[3] for word in words),
                        y1=max(word[4] for word in words),
                    ),
                )
            )
        pages[page_no] = page_lines
    return TextLayer(pages=pages, word_count=word_count, invalid_bbox_count=invalid_bbox_count)


def _intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    x = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    y = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    area = max(1e-9, left.width * left.height)
    return x * y / area


def _covers(block: Block, line: SourceLine) -> bool:
    if not block.bbox:
        return False
    cx = (line.bbox.x0 + line.bbox.x1) / 2.0
    cy = (line.bbox.y0 + line.bbox.y1) / 2.0
    inside = block.bbox.x0 <= cx <= block.bbox.x1 and block.bbox.y0 <= cy <= block.bbox.y1
    return inside or _intersection_ratio(line.bbox, block.bbox) >= 0.35


def _char_multiset_f1(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    a, b = Counter(left), Counter(right)
    matched = sum(min(count, b.get(char, 0)) for char, count in a.items())
    precision = matched / len(left)
    recall = matched / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _block_reference(block: Block, lines: list[SourceLine]) -> str:
    selected = [line for line in lines if _covers(block, line)]
    selected.sort(key=lambda line: (line.line_no, line.bbox.x0))
    return " ".join(line.text for line in selected)


def _append_unique(container: dict[str, Any], key: str, *values: str) -> None:
    existing = list(container.get(key) or [])
    container[key] = list(dict.fromkeys([*existing, *[value for value in values if value]]))


def _visible_ink(
    page_image: Path,
    bbox: BoundingBox,
    page_gray: np.ndarray | None = None,
) -> tuple[bool, float]:
    if page_gray is None:
        with Image.open(page_image) as image:
            page_gray = np.asarray(image.convert("L"))
    height, width = page_gray.shape[:2]
    left = max(0, min(width, int(bbox.x0 / 1000.0 * width)))
    top = max(0, min(height, int(bbox.y0 / 1000.0 * height)))
    right = max(0, min(width, int(np.ceil(bbox.x1 / 1000.0 * width))))
    bottom = max(0, min(height, int(np.ceil(bbox.y1 / 1000.0 * height))))
    if right - left < 2 or bottom - top < 2:
        return False, 0.0
    crop = page_gray[top:bottom, left:right]
    ink_ratio = float((crop < 210).mean()) if crop.size else 0.0
    return ink_ratio >= 0.004, ink_ratio


def _repeated_margin_lines(layer: TextLayer, page_count: int) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in layer.pages.values():
        seen = {
            line.normalized
            for line in lines
            if line.normalized and (line.bbox.y1 <= 110 or line.bbox.y0 >= 890)
        }
        counts.update(seen)
    threshold = max(3, int(page_count * 0.35))
    return {text for text, count in counts.items() if count >= threshold}


def _text_layer_reliability(document: Document, layer: TextLayer) -> dict[str, Any]:
    comparable = matched = 0
    for page in document.pages:
        lines = layer.pages.get(page.page_no, [])
        for line in lines:
            norm = line.normalized
            if len(norm) < 4:
                continue
            blocks = [
                block for block in page.blocks
                if block.block_type not in _NON_SEMANTIC_TYPES and block.content.strip() and _covers(block, line)
            ]
            if not blocks:
                continue
            comparable += 1
            if any(
                norm in normalize_text(flatten_content(block.content))
                or _char_multiset_f1(norm, normalize_text(flatten_content(block.content))) >= 0.92
                for block in blocks
            ):
                matched += 1
    ratio = matched / comparable if comparable else 0.0
    invalid_ratio = layer.invalid_bbox_count / max(1, layer.word_count + layer.invalid_bbox_count)
    reliable = layer.word_count >= 20 and comparable >= 20 and ratio >= 0.72 and invalid_ratio <= 0.02
    return {
        "word_count": layer.word_count,
        "invalid_bbox_count": layer.invalid_bbox_count,
        "invalid_bbox_ratio": round(invalid_ratio, 6),
        "comparable_line_count": comparable,
        "matched_line_count": matched,
        "agreement_ratio": round(ratio, 6),
        "reliable": reliable,
    }


def _audit_existing_blocks(
    page: Any,
    lines: list[SourceLine],
    *,
    text_layer_reliable: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for block in page.blocks:
        if not block.bbox or not block.content.strip():
            continue
        if block.block_type not in {BlockType.TEXT, BlockType.TITLE, BlockType.FOOTNOTE, BlockType.TABLE}:
            continue
        reference = _block_reference(block, lines)
        source_norm = normalize_text(flatten_content(block.content))
        reference_norm = normalize_text(reference)
        if not source_norm or not reference_norm:
            continue
        sequence = SequenceMatcher(None, source_norm, reference_norm, autojunk=False).ratio()
        char_f1 = _char_multiset_f1(source_norm, reference_norm)
        length_ratio = min(len(source_norm), len(reference_norm)) / max(len(source_norm), len(reference_norm))
        source_numbers = number_tokens(block.content)
        reference_numbers = number_tokens(reference)
        numeric_match = Counter(source_numbers) == Counter(reference_numbers)
        if block.block_type == BlockType.TABLE:
            numeric_length_ratio = min(len(source_numbers), len(reference_numbers)) / max(
                1, len(source_numbers), len(reference_numbers)
            )
            source_counter = Counter(source_numbers)
            reference_counter = Counter(reference_numbers)
            differences = list((source_counter - reference_counter).elements()) + list(
                (reference_counter - source_counter).elements()
            )
            critical_differences = [
                token for token in differences
                if "." in token or "%" in token or len(token.lstrip("+-0")) >= 2
            ]
            if (
                text_layer_reliable
                and source_numbers
                and reference_numbers
                and not numeric_match
                and numeric_length_ratio >= 0.85
                and critical_differences
            ):
                block.issue_flags = list(dict.fromkeys([*block.issue_flags, "TAB-005"]))
                _append_unique(block.metadata, "processing_tags", "pdf_text_numeric_crosscheck")
                findings.append({
                    "kind": "table_numeric_mismatch",
                    "page_no": page.page_no,
                    "block_id": block.block_id,
                    "flag": "TAB-005",
                    "mineru_numbers": source_numbers[:200],
                    "pdf_numbers": reference_numbers[:200],
                    "critical_differences": critical_differences[:100],
                })
            continue
        if source_norm == reference_norm:
            continue
        numeric_mismatch = bool(source_numbers and reference_numbers and not numeric_match)
        severe_text_mismatch = sequence < 0.82 and char_f1 < 0.90
        footnote_encoding_conflict = any(
            symbol in block.content and re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", reference)
            for symbol, letter in {"①": "a", "②": "b", "③": "c", "④": "d", "⑤": "e"}.items()
        )
        requires_visual_review = bool(
            severe_text_mismatch or numeric_mismatch or footnote_encoding_conflict
        )
        if requires_visual_review:
            block.issue_flags = list(dict.fromkeys([*block.issue_flags, "TXT-003"]))
        comparison = {
            "sequence_similarity": round(sequence, 6),
            "character_f1": round(char_f1, 6),
            "length_ratio": round(length_ratio, 6),
            "numeric_match": numeric_match,
            "footnote_encoding_conflict": footnote_encoding_conflict,
            "reference_excerpt": reference[:500],
            "text_layer_reliable": text_layer_reliable,
            "requires_visual_review": requires_visual_review,
            "mineru_preserved": True,
        }
        block.metadata["pdf_text_comparison"] = comparison
        _append_unique(block.metadata, "processing_tags", "pdf_text_block_crosscheck")
        _append_unique(block.metadata, "verification_tags", "mineru_preserved_non_destructive")
        block.revisions.append(
            Revision(
                source="pdf_text_layer_observation",
                content=reference.strip(),
                structured_data=comparison,
                prompt_version="pdf-text-crosscheck-v2-non-destructive",
            )
        )
        findings.append({
            "kind": "block_text_difference_observed",
            "page_no": page.page_no,
            "block_id": block.block_id,
            "flag": "TXT-003" if requires_visual_review else None,
            **comparison,
        })
        if requires_visual_review:
            LOGGER.info(
                "[文本层对照] page=%s block=%s decision=preserve_mineru flags=TXT-003 "
                "sequence=%.4f char_f1=%.4f length_ratio=%.4f numeric_match=%s footnote_conflict=%s",
                page.page_no,
                block.block_id,
                sequence,
                char_f1,
                length_ratio,
                numeric_match,
                footnote_encoding_conflict,
            )
    return findings


def _nearby_table(page: Any, line: SourceLine) -> Block | None:
    candidates = []
    for block in page.blocks:
        if block.block_type != BlockType.TABLE or not block.bbox:
            continue
        gap_above = block.bbox.y0 - line.bbox.y1
        gap_below = line.bbox.y0 - block.bbox.y1
        vertical_gap = max(gap_above, gap_below, 0.0)
        horizontal_overlap = max(0.0, min(block.bbox.x1, line.bbox.x1) - max(block.bbox.x0, line.bbox.x0))
        if vertical_gap <= 90 and horizontal_overlap > 0:
            candidates.append((vertical_gap, block))
    return min(candidates, key=lambda value: value[0])[1] if candidates else None


def _make_gap_block(
    document: Document,
    page: Any,
    line: SourceLine,
    *,
    text_layer_reliable: bool,
    ink_ratio: float,
) -> Block:
    table = _nearby_table(page, line)
    is_source_note = bool(re.match(r"\s*(?:数据来源|资料来源|注[：:]|说明[：:])", line.text))
    above_table = bool(table and table.bbox and line.bbox.y1 <= table.bbox.y0 + 15)
    looks_like_title = bool(
        re.search(r"(?:附表|表\s*\d|图\s*\d|图表)", line.text)
        or (table and above_table and len(line.normalized) <= 40 and line.bbox.width <= 650)
    )
    block_type = BlockType.FOOTNOTE if is_source_note else (BlockType.TITLE if looks_like_title else BlockType.TEXT)
    flags = ["TXT-004"]
    if table:
        flags.append("TAB-008")
    status = ResolutionStatus.REPAIRED if text_layer_reliable else ResolutionStatus.REVIEW
    content = line.text.strip() if text_layer_reliable else ""
    metadata: dict[str, Any] = {
        "processing_tags": ["page_reverse_audit", "visible_gap_detected"],
        "verification_tags": ["pdf_text_layer_recovered"] if text_layer_reliable else [],
        "source_line_no": line.line_no,
        "visible_ink_ratio": round(ink_ratio, 6),
        "text_layer_reliable": text_layer_reliable,
        "chunk_eligible": text_layer_reliable,
    }
    if table:
        metadata["associated_table_block_id"] = table.block_id
        _append_unique(table.metadata, "processing_tags", "table_context_recovered")
        _append_unique(table.metadata, "verification_tags", "pdf_text_layer_context")
        table.metadata.setdefault("recovered_context_block_ids", []).append("pending")
        if looks_like_title and not table.metadata.get("caption"):
            table.metadata["caption"] = line.text.strip()
    block = Block(
        document_id=document.document_id,
        page_no=page.page_no,
        block_type=block_type,
        bbox=line.bbox,
        reading_order=len(page.blocks),
        raw_content="",
        resolved_content=content,
        source="pdf_text_layer" if text_layer_reliable else "page_gap_detector",
        issue_flags=flags,
        status=status,
        metadata=metadata,
        revisions=[
            Revision(
                source="pdf_text_layer" if text_layer_reliable else "page_gap_detector",
                content=content,
                structured_data={
                    "source_line": line.text,
                    "source_line_no": line.line_no,
                    "visible_ink_ratio": round(ink_ratio, 6),
                    "text_layer_reliable": text_layer_reliable,
                },
            )
        ],
    )
    if table:
        table.metadata["recovered_context_block_ids"][-1] = block.block_id
    return block


def run_page_completeness_audit(
    document: Document,
    pdf_path: Path,
    work_dir: Path,
    *,
    render_dpi: int = 220,
) -> dict[str, Any]:
    """Reverse-audit every PDF page before block-level VLM routing.

    A reliable text layer is used as a low-cost second source.  It can recover
    a block only when its glyph region is visibly non-empty and no MinerU
    semantic block carries the same content.  Unreliable text layers create a
    review candidate with no guessed text, which is then routed to VLM.
    """
    output_dir = work_dir / "postprocess"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = extract_pdf_text_layer(pdf_path)
    reliability = _text_layer_reliability(document, layer)
    repeated = _repeated_margin_lines(layer, document.page_count)
    findings: list[dict[str, Any]] = []
    recovered = 0
    review_candidates = 0

    for page in document.pages:
        lines = layer.pages.get(page.page_no, [])
        findings.extend(
            _audit_existing_blocks(
                page,
                lines,
                text_layer_reliable=bool(reliability["reliable"]),
            )
        )
        page_text = normalize_text(" ".join(flatten_content(block.content) for block in page.blocks))
        page_image: Path | None = None
        page_gray: np.ndarray | None = None
        additions: list[Block] = []
        empty_block_lines: dict[str, tuple[Block, list[SourceLine], list[float]]] = {}
        for line in lines:
            norm = line.normalized
            if len(norm) < 2 or norm in repeated or norm in page_text:
                continue
            covering = [block for block in page.blocks if _covers(block, line)]
            visual_covering = [
                block for block in covering if block.block_type in {BlockType.IMAGE, BlockType.CHART}
            ]
            if visual_covering:
                for visual in visual_covering:
                    _append_unique(visual.metadata, "processing_tags", "pdf_text_visible_inside_visual")
                findings.append({
                    "kind": "visual_region_text",
                    "page_no": page.page_no,
                    "block_ids": [block.block_id for block in visual_covering],
                    "source_text": line.text,
                    "bbox": line.bbox.as_list(),
                })
                continue
            semantic_covering = [
                block for block in covering
                if block.block_type not in _NON_SEMANTIC_TYPES and block.content.strip()
            ]
            if semantic_covering:
                continue
            if page_image is None:
                page_image = render_page(pdf_path, page.page_no, work_dir / "pages", dpi=render_dpi)
                with Image.open(page_image) as image:
                    page_gray = np.asarray(image.convert("L"))
            visible, ink_ratio = _visible_ink(page_image, line.bbox, page_gray)
            if not visible:
                continue
            empty_semantic = [
                block for block in covering
                if block.block_type not in _NON_SEMANTIC_TYPES and not block.content.strip()
            ]
            if empty_semantic:
                target = max(
                    empty_semantic,
                    key=lambda block: _intersection_ratio(line.bbox, block.bbox) if block.bbox else 0.0,
                )
                current = empty_block_lines.setdefault(target.block_id, (target, [], []))
                current[1].append(line)
                current[2].append(ink_ratio)
                continue
            block = _make_gap_block(
                document,
                page,
                line,
                text_layer_reliable=bool(reliability["reliable"]),
                ink_ratio=ink_ratio,
            )
            additions.append(block)
            if block.status == ResolutionStatus.REPAIRED:
                recovered += 1
            else:
                review_candidates += 1
            findings.append({
                "kind": "missing_block",
                "page_no": page.page_no,
                "block_id": block.block_id,
                "flag": block.issue_flags,
                "bbox": line.bbox.as_list(),
                "source_text": line.text,
                "status": block.status.value,
                "visible_ink_ratio": round(ink_ratio, 6),
            })
        for target, target_lines, ink_ratios in empty_block_lines.values():
            target_lines.sort(key=lambda value: (value.line_no, value.bbox.x0))
            recovered_text = "\n".join(line.text for line in target_lines)
            reliable = bool(reliability["reliable"])
            target.resolved_content = recovered_text if reliable else ""
            target.source = "pdf_text_layer" if reliable else "page_gap_detector"
            target.status = ResolutionStatus.REPAIRED if reliable else ResolutionStatus.REVIEW
            target.issue_flags = list(dict.fromkeys([*target.issue_flags, "TXT-004"]))
            target.metadata.update(
                {
                    "text_layer_reliable": reliable,
                    "chunk_eligible": reliable,
                    "visible_ink_ratio": round(max(ink_ratios), 6),
                    "source_line_numbers": [line.line_no for line in target_lines],
                }
            )
            _append_unique(target.metadata, "processing_tags", "page_reverse_audit", "empty_block_recovered")
            if reliable:
                _append_unique(target.metadata, "verification_tags", "pdf_text_layer_recovered")
                recovered += 1
            else:
                review_candidates += 1
            target.revisions.append(
                Revision(
                    source=target.source,
                    content=target.resolved_content,
                    structured_data={
                        "source_text": recovered_text,
                        "source_line_numbers": target.metadata["source_line_numbers"],
                        "text_layer_reliable": reliable,
                    },
                )
            )
            findings.append(
                {
                    "kind": "empty_block_recovered",
                    "page_no": page.page_no,
                    "block_id": target.block_id,
                    "flag": ["TXT-004"],
                    "source_text": recovered_text,
                    "status": target.status.value,
                    "bbox": target.bbox.as_list() if target.bbox else None,
                }
            )
        page.blocks.extend(additions)
        page.metadata["page_completeness"] = {
            "source_line_count": len(lines),
            "new_block_count": len(additions),
            "text_layer_reliable": bool(reliability["reliable"]),
        }

    summary = {
        "text_layer": reliability,
        "repeated_margin_line_count": len(repeated),
        "finding_count": len(findings),
        "recovered_block_count": recovered,
        "review_candidate_count": review_candidates,
        "findings_by_kind": dict(sorted(Counter(item["kind"] for item in findings).items())),
        "findings_path": str(output_dir / "page_audit.jsonl"),
    }
    with (output_dir / "page_audit.jsonl").open("w", encoding="utf-8") as handle:
        for finding in findings:
            handle.write(json.dumps(finding, ensure_ascii=False) + "\n")
    (output_dir / "page_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
