from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from mineru_vlm_rag.domain.models import (
    Asset,
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
    ResolutionStatus,
    Revision,
)


TYPE_MAP = {
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "title": BlockType.TITLE,
    "table": BlockType.TABLE,
    "image": BlockType.IMAGE,
    "chart": BlockType.CHART,
    "equation": BlockType.FORMULA,
    "formula": BlockType.FORMULA,
    "interline_equation": BlockType.FORMULA,
    "seal": BlockType.SEAL,
    "footnote": BlockType.FOOTNOTE,
    "page_footnote": BlockType.FOOTNOTE,
    "aside_text": BlockType.TEXT,
    "header": BlockType.HEADER,
    "footer": BlockType.FOOTER,
    "page_number": BlockType.PAGE_NUMBER,
}


def _find_content_list(extracted_dir: Path) -> Path:
    candidates = sorted(extracted_dir.rglob("*_content_list.json"))
    if not candidates:
        candidates = sorted(extracted_dir.rglob("content_list.json"))
    if not candidates:
        raise FileNotFoundError("MinerU archive does not contain content_list JSON")
    return candidates[0]


def _read_content(item: dict[str, Any], block_type: BlockType) -> str:
    if block_type == BlockType.TABLE:
        return str(item.get("table_body") or item.get("html") or item.get("text") or "")
    if block_type == BlockType.FORMULA:
        return str(item.get("latex") or item.get("text") or "")
    if block_type in (BlockType.IMAGE, BlockType.CHART):
        caption = item.get("image_caption") or item.get("caption") or []
        if isinstance(caption, list):
            caption = "\n".join(str(value) for value in caption)
        return str(caption or item.get("text") or "")
    return str(item.get("text") or item.get("content") or "")


def _asset_from_item(
    item: dict[str, Any],
    extracted_dir: Path,
    document_id: str,
    page_no: int,
    kind: str,
) -> Asset | None:
    raw_path = item.get("img_path") or item.get("image_path")
    if not raw_path:
        return None
    matches = list(extracted_dir.rglob(Path(str(raw_path)).name))
    if not matches:
        return None
    path = matches[0]
    data = path.read_bytes()
    try:
        with Image.open(path) as image:
            width, height = image.size
            mime = Image.MIME.get(image.format or "", "image/png")
    except Exception:
        width = height = None
        mime = "application/octet-stream"
    return Asset(
        document_id=document_id,
        page_no=page_no,
        kind=kind,
        mime_type=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        data=data,
        source_path=str(path),
    )


def parse_mineru_output(document: Document, extracted_dir: Path) -> Document:
    content_path = _find_content_list(extracted_dir)
    payload = json.loads(content_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("content_list") or payload.get("data") or []
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("Unexpected MinerU content_list structure")

    pages = {index: Page(document_id=document.document_id, page_no=index) for index in range(1, document.page_count + 1)}
    assets: list[Asset] = []
    per_page_order: dict[int, int] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        page_no = int(item.get("page_idx", item.get("page_no", 0))) + (1 if "page_idx" in item else 0)
        page_no = max(1, page_no)
        if page_no not in pages:
            pages[page_no] = Page(document_id=document.document_id, page_no=page_no)
        page_size = item.get("page_size")
        if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
            pages[page_no].metadata["mineru_coordinate_width"] = float(page_size[0])
            pages[page_no].metadata["mineru_coordinate_height"] = float(page_size[1])
            pages[page_no].metadata["mineru_coordinate_space"] = "explicit_page_size"
        raw_type = str(item.get("type") or item.get("block_type") or "unknown").lower()
        block_type = TYPE_MAP.get(raw_type, BlockType.UNKNOWN)
        content = _read_content(item, block_type)
        asset = _asset_from_item(item, extracted_dir, document.document_id, page_no, block_type.value)
        if asset:
            assets.append(asset)
        order = per_page_order.get(page_no, 0)
        per_page_order[page_no] = order + 1
        block = Block(
            document_id=document.document_id,
            page_no=page_no,
            block_type=block_type,
            bbox=BoundingBox.from_sequence(item.get("bbox")),
            reading_order=order,
            raw_content=content,
            resolved_content=content,
            asset_id=asset.asset_id if asset else None,
            metadata={
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {"text", "content", "table_body"}
                },
                # Keep MinerU's sequence immutable.  Local layout analysis may
                # derive a different order, but must never destroy the only
                # upstream ordering evidence available for later review.
                "mineru_order": order,
                "mineru_raw_type": raw_type,
            },
            revisions=[Revision(source="mineru", content=content, structured_data={"raw_type": raw_type})],
            status=ResolutionStatus.ACCEPTED,
        )
        pages[page_no].blocks.append(block)

    for page in pages.values():
        boxes = [block.bbox for block in page.blocks if block.bbox is not None]
        if boxes:
            page.metadata["mineru_bbox_max_x"] = max(max(box.x0, box.x1) for box in boxes)
            page.metadata["mineru_bbox_max_y"] = max(max(box.y0, box.y1) for box in boxes)
        if boxes and "mineru_coordinate_width" not in page.metadata:
            # MinerU API/VLM content_list output uses a normalized 0..1000
            # square when no per-item page_size is supplied.  PDF point sizes
            # are a different coordinate space and must not be substituted.
            page.metadata["mineru_coordinate_width"] = 1000.0
            page.metadata["mineru_coordinate_height"] = 1000.0
            page.metadata["mineru_coordinate_space"] = "normalized_1000_inferred"

    document.pages = [pages[index] for index in sorted(pages)]
    document.assets = assets
    return document
