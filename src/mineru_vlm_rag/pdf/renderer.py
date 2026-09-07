from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

from mineru_vlm_rag.domain.models import BoundingBox


class PDFRenderError(RuntimeError):
    pass


def render_page(pdf_path: Path, page_no: int, output_dir: Path, dpi: int = 220) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"page-{page_no:04d}"
    output_path = prefix.with_suffix(".png")
    if output_path.exists():
        return output_path
    command = [
        "pdftoppm",
        "-f", str(page_no),
        "-l", str(page_no),
        "-r", str(dpi),
        "-singlefile",
        "-png",
        str(pdf_path),
        str(prefix),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise PDFRenderError(f"pdftoppm failed on page {page_no}: {result.stderr.strip()}")
    return output_path


def render_block(
    page_image: Path,
    bbox: BoundingBox | None,
    source_width: float | None,
    source_height: float | None,
    output_path: Path,
    padding: int = 12,
    *,
    crop_details: dict[str, Any] | None = None,
) -> Path:
    """Render a block crop, falling back to the full page for unusable boxes.

    MinerU variants do not all expose the same coordinate metadata.  A bad or
    mismatched coordinate space must never abort the document: the VLM can
    still inspect the full page, and ``crop_details`` records that decision.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_image) as image:
        details: dict[str, Any] = {
            "page_image_size": [image.width, image.height],
            "source_coordinate_size": [source_width, source_height],
            "bbox": bbox.as_list() if bbox else None,
        }
        source_size_is_valid = (
            source_width is not None
            and source_height is not None
            and math.isfinite(source_width)
            and math.isfinite(source_height)
            and source_width > 0
            and source_height > 0
        )
        if bbox is None:
            crop = image.copy()
            details.update(mode="full_page_no_bbox", reason="missing_bbox")
        elif not source_size_is_valid:
            crop = image.copy()
            details.update(mode="full_page_fallback", reason="invalid_source_coordinate_size")
        else:
            coordinates = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            if not all(math.isfinite(value) for value in coordinates):
                crop = image.copy()
                details.update(mode="full_page_fallback", reason="non_finite_bbox")
            else:
                x0, x1 = sorted((bbox.x0, bbox.x1))
                y0, y1 = sorted((bbox.y0, bbox.y1))
                sx = image.width / source_width
                sy = image.height / source_height
                left = max(0, min(image.width, math.floor(x0 * sx) - padding))
                top = max(0, min(image.height, math.floor(y0 * sy) - padding))
                right = max(0, min(image.width, math.ceil(x1 * sx) + padding))
                bottom = max(0, min(image.height, math.ceil(y1 * sy) + padding))
                pixel_box = [left, top, right, bottom]
                details["pixel_bbox"] = pixel_box
                if right - left < 2 or bottom - top < 2:
                    crop = image.copy()
                    details.update(mode="full_page_fallback", reason="bbox_outside_or_degenerate")
                else:
                    crop = image.crop(tuple(pixel_box))
                    details.update(mode="bbox_crop", reason=None)
        crop.save(output_path, format="PNG")
        details["output_size"] = [crop.width, crop.height]
        if crop_details is not None:
            crop_details.update(details)
    return output_path


def enhance_image(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        enhanced = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.5)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        enhanced.save(output_path, format="PNG")
    return output_path
