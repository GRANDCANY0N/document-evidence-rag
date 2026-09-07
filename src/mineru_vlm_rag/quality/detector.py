from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
from lxml import html

from mineru_vlm_rag.domain.models import Block, BlockType
from mineru_vlm_rag.tables import canonicalize_table


@dataclass(frozen=True)
class DetectionContext:
    repeated_char_ratio: float = 0.45
    blur_variance_threshold: float = 80.0
    low_contrast_std_threshold: float = 22.0
    dark_region_ratio: float = 0.25


VLM_FLAGS = {
    "SRC-002", "SRC-003", "LAY-002", "LAY-004",
    "TXT-001", "TXT-002", "TXT-003", "TAB-001", "TAB-002",
    "TAB-003", "TAB-004", "TAB-005", "TAB-006", "TAB-007",
    "TAB-009", "TAB-010", "TXT-004",
    "IMG-001", "IMG-002", "IMG-003", "FOR-001", "FOR-002", "FOR-003",
}


def image_quality_flags(path: Path, context: DetectionContext) -> list[str]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    flags: list[str] = []
    variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    if variance < context.blur_variance_threshold:
        flags.append("SRC-003")
    if float(image.std()) < context.low_contrast_std_threshold:
        flags.append("SRC-002")
    dark_ratio = float((image < 20).mean())
    if dark_ratio >= context.dark_region_ratio:
        flags.append("SRC-004")
    return flags


def _text_flags(content: str, context: DetectionContext) -> list[str]:
    flags: list[str] = []
    compact = re.sub(r"\s+", "", content)
    if not compact:
        return flags
    if re.search(r"(?:█{3,}|\*{5,}|已脱敏|已遮挡|redacted)", compact, flags=re.I):
        flags.append("SRC-004")
    if len(compact) >= 8:
        most_common = Counter(compact).most_common(1)[0][1] / len(compact)
        if most_common >= context.repeated_char_ratio:
            flags.append("TXT-002")
    if "�" in content or re.search(r"[?？]{4,}", content):
        flags.append("TXT-001")
    if re.search(r"(?:\d+[A-Za-z]+|[A-Za-z]+\d+).*[¥￥$]|[¥￥$].*(?:\d+[A-Za-z]+)", content):
        flags.append("TXT-002")
    return flags


def _table_flags(content: str) -> list[str]:
    flags: list[str] = []
    if not content.strip():
        return ["TAB-001"]
    try:
        table = canonicalize_table(content)
        if table.height < 1 or table.width < 1:
            flags.append("TAB-001")
    except Exception:
        flags.append("TAB-002")
    return flags


def _formula_flags(content: str) -> list[str]:
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    if not content.strip() or any(content.count(left) != content.count(right) for left, right in pairs):
        return ["FOR-001"]
    return []


def detect_block_issues(
    block: Block,
    context: DetectionContext,
    asset_path: Path | None = None,
) -> list[str]:
    flags: list[str] = []
    if block.block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.FOOTNOTE}:
        flags.extend(_text_flags(block.content, context))
    elif block.block_type == BlockType.TABLE:
        flags.extend(_table_flags(block.content))
    elif block.block_type in {BlockType.IMAGE, BlockType.CHART}:
        flags.append("IMG-002" if block.block_type == BlockType.CHART else "IMG-001")
        if not block.content.strip():
            flags.append("IMG-003")
    elif block.block_type == BlockType.FORMULA:
        flags.extend(_formula_flags(block.content))
    elif block.block_type == BlockType.SEAL:
        flags.append("LAY-004")
    if asset_path and asset_path.exists():
        flags.extend(image_quality_flags(asset_path, context))
    return list(dict.fromkeys(flags))


def needs_vlm(flags: list[str]) -> bool:
    return any(flag in VLM_FLAGS for flag in flags) and "SRC-004" not in flags
