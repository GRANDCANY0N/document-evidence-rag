from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class OcclusionRegion:
    x: int
    y: int
    width: int
    height: int
    fill_ratio: float


def detect_solid_occlusions(
    image_path: Path,
    *,
    dark_threshold: int = 18,
    minimum_area_ratio: float = 0.0005,
    minimum_fill_ratio: float = 0.9,
    minimum_aspect_ratio: float = 3.0,
) -> list[OcclusionRegion]:
    """Find solid dark redaction bars while rejecting ordinary glyphs and chart strokes."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    mask = (image <= dark_threshold).astype("uint8")
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = image.shape[0] * image.shape[1] * minimum_area_ratio
    regions: list[OcclusionRegion] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if width < 40 or height < 8 or area < minimum_area:
            continue
        box_area = width * height
        fill_ratio = area / box_area if box_area else 0.0
        aspect = max(width / max(height, 1), height / max(width, 1))
        if fill_ratio < minimum_fill_ratio or aspect < minimum_aspect_ratio:
            continue
        regions.append(OcclusionRegion(x=x, y=y, width=width, height=height, fill_ratio=fill_ratio))
    return sorted(regions, key=lambda value: (value.y, value.x))
