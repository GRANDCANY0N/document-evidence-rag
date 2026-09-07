from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lxml import etree, html
from PIL import Image


def _region(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.01:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    left, right = sorted((max(0, int(x0)), min(width, int(x1))))
    top, bottom = sorted((max(0, int(y0)), min(height, int(y1))))
    if right - left < 10 or bottom - top < 10:
        return None
    return left, top, right, bottom


def _grid_rotation_override(image: Image.Image) -> int:
    """Detect a sideways ruled table when the VLM orientation answer is 0."""
    if image.height <= image.width * 1.2:
        return 0
    gray = np.asarray(image.convert("L"))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, image.width // 18), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, image.height // 18))),
    )
    horizontal_lines = sum(
        cv2.boundingRect(contour)[2] >= image.width * 0.55
        for contour in cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    )
    vertical_lines = sum(
        cv2.boundingRect(contour)[3] >= image.height * 0.55
        for contour in cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    )
    return 90 if horizontal_lines >= 3 and horizontal_lines > max(1, vertical_lines) * 1.4 else 0


def _ruled_header_bottom(image: Image.Image) -> int | None:
    gray = np.asarray(image.convert("L"))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, image.width // 8), 1)),
    )
    positions: list[int] = []
    for contour in cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= image.width * 0.55:
            positions.append(y + height // 2)
    upper_limit = max(80, int(image.height * 0.18))
    candidates = sorted({position for position in positions if 5 <= position <= upper_limit})
    return candidates[-1] if candidates else None


def _bright_horizontal_boundaries(image: Image.Image) -> list[int]:
    """Find light separator rules used by coloured government-report tables."""
    gray = np.asarray(image.convert("L"))
    bright_fraction = (gray >= 245).mean(axis=1)
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    candidate_rows = np.flatnonzero(
        (bright_fraction >= 0.80) | ((row_mean >= 235) & (row_std <= 4.0))
    ).tolist()
    if not candidate_rows:
        return []
    clusters: list[list[int]] = [[candidate_rows[0]]]
    for row in candidate_rows[1:]:
        if row <= clusters[-1][-1] + 1:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    # Ignore broad white page margins; a separator is normally only a few
    # pixels high after rasterisation.
    return [
        int(round(sum(cluster) / len(cluster)))
        for cluster in clusters
        if len(cluster) <= max(8, image.height // 100)
    ]


def _row_aligned_ranges(
    data_top: int,
    data_bottom: int,
    available: int,
    boundaries: list[int],
) -> list[tuple[int, int]]:
    """Choose tile cuts on detected row rules and overlap one complete row."""
    usable = sorted({value for value in boundaries if data_top < value < data_bottom})
    if len(usable) < 3:
        return []
    ranges: list[tuple[int, int]] = []
    start = data_top
    while start < data_bottom:
        target = start + available
        if target >= data_bottom - max(12, int(available * 0.03)):
            ranges.append((start, data_bottom))
            break
        candidates = [value for value in usable if start + 40 < value <= target]
        if not candidates:
            return []
        end = candidates[-1]
        ranges.append((start, end))
        earlier = [value for value in usable if start < value < end]
        next_start = earlier[-1] if earlier else end
        if next_start <= start:
            return []
        start = next_start
    return ranges


def create_table_tiles(
    image_path: Path,
    output_dir: Path,
    structure: dict[str, Any],
    *,
    max_tile_height: int = 1400,
    overlap: int = 100,
) -> list[Path]:
    """Split only the data area and repeat the detected header on every tile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        try:
            rotation = int(structure.get("rotation_degrees") or 0)
        except (TypeError, ValueError):
            rotation = 0
        rotation_overridden = False
        if rotation == 0:
            detected_rotation = _grid_rotation_override(image)
            if detected_rotation:
                rotation = detected_rotation
                rotation_overridden = True
        if rotation in {90, 180, 270}:
            image = image.rotate(-rotation, expand=True, fillcolor="white")
        structure["effective_rotation_degrees"] = rotation
        structure["rotation_overridden_by_grid"] = rotation_overridden
        structure["oriented_image_size"] = [image.width, image.height]
        image.save(output_dir / "oriented-table.png", format="PNG")
        bright_boundaries = _bright_horizontal_boundaries(image)
        header_box = None if rotation_overridden else _region(
            structure.get("header_region"), image.width, image.height
        )
        data_box = None if rotation_overridden else _region(
            structure.get("data_region"), image.width, image.height
        )
        data_was_missing = data_box is None
        if data_box is None:
            # Conservative fallback: preserve the whole table.  The default
            # header band is repeated only when more than one tile is needed.
            data_box = (0, 0, image.width, image.height)
        if header_box is None:
            ruled_bottom = _ruled_header_bottom(image)
            estimated = min(ruled_bottom or max(80, int(image.height * 0.16)), data_box[3])
            header_box = (data_box[0], data_box[1], data_box[2], estimated)
        bright_header_candidates = [
            value for value in bright_boundaries if 8 <= value <= int(image.height * 0.22)
        ]
        if bright_header_candidates:
            # The first full-width light rule after a coloured header is more
            # precise than an advisory VLM region and avoids cutting row 1.
            bright_header_bottom = bright_header_candidates[0]
            header_box = (0, 0, image.width, bright_header_bottom)
            data_box = (0, bright_header_bottom, image.width, data_box[3])
        if data_was_missing:
            data_box = (0, min(header_box[3], image.height - 1), image.width, image.height)
        # VLM coordinates are advisory. A data region covering only a small
        # part of a long image would silently drop lower rows. Preserve all
        # pixels below the header in that case; overlap handles boundary rows.
        if image.height > max_tile_height and data_box[3] - data_box[1] < image.height * 0.55:
            data_box = (0, min(header_box[3], image.height - 1), image.width, image.height)
        if bright_boundaries and data_box[3] < image.height - 20:
            last_rule = max(bright_boundaries)
            lower = np.asarray(image.convert("L"))[data_box[3]:]
            lower_ink_ratio = float((lower < 245).mean()) if lower.size else 0.0
            if last_rule >= data_box[3] - max(20, image.height // 50) and lower_ink_ratio > 0.02:
                # A predicted data bottom landing on/just before an internal
                # row rule would truncate the final row. Preserve the tail.
                data_box = (data_box[0], data_box[1], data_box[2], image.height)
        maximum_header_height = max(80, int(max_tile_height * 0.42))
        if header_box[3] - header_box[1] > maximum_header_height:
            header_box = (
                header_box[0],
                header_box[1],
                header_box[2],
                header_box[1] + maximum_header_height,
            )
        header = image.crop(header_box)
        data_left, data_top, data_right, data_bottom = data_box
        available = max(200, max_tile_height - header.height)
        if data_bottom - data_top <= max_tile_height:
            target = output_dir / "tile-0001.png"
            image.crop(data_box).save(target, format="PNG")
            return [target]
        aligned_ranges = _row_aligned_ranges(
            data_top,
            data_bottom,
            available,
            bright_boundaries,
        )
        step = max(1, available - overlap)
        paths: list[Path] = []
        start = data_top
        index = 1
        while start < data_bottom:
            if aligned_ranges:
                start, end = aligned_ranges[index - 1]
            else:
                end = min(data_bottom, start + available)
            data = image.crop((data_left, start, data_right, end))
            canvas = Image.new("RGB", (max(header.width, data.width), header.height + data.height), "white")
            canvas.paste(header, (0, 0))
            canvas.paste(data, (0, header.height))
            target = output_dir / f"tile-{index:04d}.png"
            canvas.save(target, format="PNG")
            paths.append(target)
            if end >= data_bottom:
                break
            if aligned_ranges:
                if index >= len(aligned_ranges):
                    break
                start = aligned_ranges[index][0]
            else:
                start += step
            index += 1
        return paths


def merge_table_fragments(
    fragments: list[str],
    *,
    header_rows: list[int] | None = None,
) -> str:
    """Merge table fragments and remove exact overlap/header duplicates."""
    parsed = []
    for fragment in fragments:
        root = html.fromstring(fragment)
        table = root if root.tag.lower() == "table" else (root.xpath(".//table") or [None])[0]
        if table is None:
            continue
        # Models occasionally emit rowspan=3 while producing only two thead
        # rows. Cap spans at the actual header boundary so the first data row
        # is not shifted by a phantom header cell.
        table_headers = table.xpath("./thead/tr")
        for row_index, row in enumerate(table_headers):
            maximum_span = max(1, len(table_headers) - row_index)
            for cell in row.xpath("./th|./td"):
                try:
                    rowspan = int(cell.get("rowspan", "1"))
                except ValueError:
                    rowspan = 1
                if rowspan > maximum_span:
                    cell.set("rowspan", str(maximum_span))
        parsed.append(table)
    if not parsed:
        return ""
    def row_text(row: Any) -> str:
        return "|".join(" ".join("".join(cell.itertext()).split()) for cell in row.xpath("./th|./td"))

    def split_rows(table: Any, index: int) -> tuple[list[Any], list[Any]]:
        explicit_headers = table.xpath("./thead/tr")
        if explicit_headers:
            body = table.xpath("./tbody/tr")
            if not body:
                body = [row for row in table.xpath(".//tr") if row not in explicit_headers]
            return explicit_headers, body
        rows = table.xpath("./tbody/tr") or table.xpath("./tr") or table.xpath(".//tr")
        hint = max(0, int(header_rows[index])) if header_rows and index < len(header_rows) else 1
        return rows[:hint], rows[hint:]

    first_headers, first_body = split_rows(parsed[0], 0)
    base = etree.Element("table")
    thead = etree.SubElement(base, "thead")
    tbody = etree.SubElement(base, "tbody")
    for row in first_headers:
        thead.append(row)
    seen: set[str] = set()
    for row in first_body:
        value = row_text(row)
        if value and value not in seen:
            tbody.append(row)
            seen.add(value)
    for index, table in enumerate(parsed[1:], start=1):
        _, rows = split_rows(table, index)
        for row in rows:
            value = row_text(row)
            if not value or value in seen:
                continue
            tbody.append(row)
            seen.add(value)
    return etree.tostring(base, encoding="unicode", method="html")
