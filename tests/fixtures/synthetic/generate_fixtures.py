#!/usr/bin/env python3
"""Generate deterministic, synthetic PDF fixtures for document-understanding tests.

The generator is deliberately self-contained apart from Pillow.  It draws every
page as a raster image and embeds the image with a tiny deterministic PDF writer.
No network, office suite, database, or project configuration is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PORTRAIT = (1190, 1684)
LANDSCAPE = (1684, 1190)
WHITE = (255, 255, 255)
INK = (24, 31, 40)
MUTED = (91, 101, 114)
BLUE = (31, 86, 168)
LIGHT_BLUE = (226, 238, 252)
LIGHT_GRAY = (239, 242, 245)
RED = (168, 45, 45)
GREEN = (42, 126, 78)
AMBER = (203, 132, 22)

FONT_CANDIDATES = {
    "sans": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ),
    "bold": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ),
    "serif": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ),
    "mono": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ),
    "cjk": (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ),
}


def resolve_font(kind: str) -> str:
    for candidate in FONT_CANDIDATES[kind]:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(f"No usable {kind!r} font found; checked {FONT_CANDIDATES[kind]}")


FONT_PATHS = {kind: resolve_font(kind) for kind in FONT_CANDIDATES}


def font(size: int, kind: str = "sans") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATHS[kind], size=size)


def page(size: tuple[int, int] = PORTRAIT, color: tuple[int, int, int] = WHITE) -> Image.Image:
    return Image.new("RGB", size, color)


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> float:
    box = draw.textbbox((0, 0), text, font=text_font)
    return float(box[2] - box[0])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        proposed = f"{current} {word}"
        if text_width(draw, proposed, text_font) <= width:
            current = proposed
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    width: int,
    *,
    fill: tuple[int, int, int] = INK,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    lines = wrap_text(draw, text, text_font, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    x, y = xy
    line_height = text_font.size + line_gap
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height
    return y


def draw_page_heading(
    image: Image.Image,
    title: str,
    subtitle: str,
    *,
    page_label: str,
) -> tuple[ImageDraw.ImageDraw, int]:
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    draw.rectangle((0, 0, width, 20), fill=BLUE)
    draw.text((78, 58), title, font=font(42, "bold"), fill=INK)
    draw.text((80, 116), subtitle, font=font(22), fill=MUTED)
    label_font = font(19, "mono")
    label_width = text_width(draw, page_label, label_font)
    draw.text((width - 78 - label_width, 75), page_label, font=label_font, fill=BLUE)
    draw.line((78, 158, width - 78, 158), fill=(194, 202, 211), width=2)
    return draw, 190


def draw_token_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    token: str,
    body: str,
    *,
    fill: tuple[int, int, int] = (249, 250, 252),
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=(173, 183, 194), width=2)
    draw.rounded_rectangle((x0 + 16, y0 + 14, x0 + 128, y0 + 52), radius=8, fill=BLUE)
    draw.text((x0 + 28, y0 + 19), token, font=font(20, "bold"), fill=WHITE)
    draw_wrapped(draw, (x0 + 18, y0 + 68), body, font(24), x1 - x0 - 36, line_gap=9)


def two_column_pages() -> list[Image.Image]:
    image = page()
    draw, top = draw_page_heading(
        image,
        "Two-column reading order",
        "Expected order: title, left column top-to-bottom, then right column top-to-bottom.",
        page_label="2COL-P1",
    )
    margin = 78
    gutter = 48
    col_width = (image.width - 2 * margin - gutter) // 2
    left_x = margin
    right_x = margin + col_width + gutter
    draw.rectangle((left_x, top, left_x + col_width, top + 48), fill=LIGHT_BLUE)
    draw.rectangle((right_x, top, right_x + col_width, top + 48), fill=LIGHT_BLUE)
    draw.text((left_x + 14, top + 9), "LEFT COLUMN", font=font(24, "bold"), fill=BLUE)
    draw.text((right_x + 14, top + 9), "RIGHT COLUMN", font=font(24, "bold"), fill=BLUE)
    bodies = {
        "L1": "Aster begins the first narrative. It must precede every token in the right column.",
        "L2": "Birch continues underneath Aster. The gutter is intentionally wide and empty.",
        "L3": "Cedar closes the left column before reading jumps back to the top on the right.",
        "R1": "Dahlia begins only after Cedar. A geometric reading order would incorrectly interleave it.",
        "R2": "Elm follows Dahlia. Each block carries a stable identifier for exact scoring.",
        "R3": "Fir is the final block on this page. Expected sequence ends with token R3.",
    }
    y_positions = [top + 72, top + 396, top + 720]
    for token, y in zip(("L1", "L2", "L3"), y_positions):
        draw_token_box(draw, (left_x, y, left_x + col_width, y + 270), token, bodies[token])
    for token, y in zip(("R1", "R2", "R3"), y_positions):
        draw_token_box(draw, (right_x, y, right_x + col_width, y + 270), token, bodies[token])
    draw.text(
        (margin, 1608),
        "Ground truth token order: DOC_TITLE > L1 > L2 > L3 > R1 > R2 > R3 > CAPTION",
        font=font(18, "mono"),
        fill=MUTED,
    )
    return [image]


def mixed_orientation_pages() -> list[Image.Image]:
    portrait = page()
    draw, top = draw_page_heading(
        portrait,
        "Mixed horizontal and vertical text",
        "Horizontal English and vertical Chinese coexist on a portrait page.",
        page_label="MIX-P1",
    )
    draw.rounded_rectangle((78, top, 900, 1430), radius=14, outline=(165, 176, 188), width=2)
    y = top + 36
    paragraphs = [
        ("H1", "Horizontal section one: the vessel departed at 08:30 and reached checkpoint ALPHA."),
        ("H2", "Horizontal section two: sample labels remain upright and preserve normal word spacing."),
        ("H3", "Horizontal section three: the vertical sidebar is a separate reading region."),
    ]
    for token, body in paragraphs:
        draw.rounded_rectangle((112, y, 184, y + 42), radius=7, fill=BLUE)
        draw.text((124, y + 8), token, font=font(19, "bold"), fill=WHITE)
        y = draw_wrapped(draw, (112, y + 62), body, font(29), 730, line_gap=14) + 72
    draw.rectangle((938, top, 1112, 1430), fill=(248, 244, 232), outline=AMBER, width=2)
    vertical = "纵向文字测试区"
    char_font = font(39, "cjk")
    y = top + 42
    for char in vertical:
        draw.text((1000, y), char, font=char_font, fill=INK)
        y += 70
    draw.text((965, 1330), "V1", font=font(24, "bold"), fill=RED)

    landscape = page(LANDSCAPE)
    draw, top = draw_page_heading(
        landscape,
        "Landscape page inside portrait document",
        "This second page changes MediaBox orientation and contains rotated labels.",
        page_label="MIX-P2",
    )
    cx, cy = landscape.width // 2, (top + landscape.height - 90) // 2
    draw.rounded_rectangle((110, top + 30, landscape.width - 110, landscape.height - 100), radius=20, fill=(248, 250, 253), outline=BLUE, width=3)
    draw.line((cx, top + 70, cx, landscape.height - 150), fill=(175, 185, 196), width=3)
    draw.text((160, top + 95), "LANDSCAPE-H1", font=font(38, "bold"), fill=BLUE)
    draw_wrapped(
        draw,
        (160, top + 165),
        "A wide page tests mixed page orientation. The expected page order remains portrait then landscape.",
        font(31),
        550,
        line_gap=14,
    )
    strip = Image.new("RGBA", (500, 90), (0, 0, 0, 0))
    strip_draw = ImageDraw.Draw(strip)
    strip_draw.rounded_rectangle((0, 0, 500, 90), radius=12, fill=(225, 237, 251, 255))
    strip_draw.text((24, 18), "ROTATED-90-DEGREES", font=font(29, "bold"), fill=BLUE)
    strip = strip.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    landscape.paste(strip, (cx + 300, cy - strip.height // 2), strip)
    draw.text((cx + 100, top + 100), "LANDSCAPE-H2", font=font(38, "bold"), fill=GREEN)
    draw_wrapped(
        draw,
        (cx + 100, top + 175),
        "The narrow vertical badge is rotated, while this paragraph remains horizontal.",
        font(31),
        520,
        line_gap=14,
    )
    return [portrait, landscape]


TABLE_HEADERS = ("Row ID", "Batch", "Item", "Qty", "Unit price", "Status")


def draw_cross_page_table(image: Image.Image, logical_page: int, rows: Sequence[tuple[str, ...]]) -> None:
    draw, top = draw_page_heading(
        image,
        "Cross-page table with repeated header",
        f"Compact boundary surrogate: physical page maps to logical page {logical_page} of 66.",
        page_label=f"XPT-L{logical_page:02d}",
    )
    widths = [150, 150, 260, 120, 160, 190]
    x0 = 80
    header_h = 74
    row_h = 116
    y = top + 25
    draw.rectangle((x0, y, x0 + sum(widths), y + header_h), fill=(42, 82, 130), outline=INK, width=2)
    x = x0
    for value, width in zip(TABLE_HEADERS, widths):
        draw.text((x + 12, y + 22), value, font=font(22, "bold"), fill=WHITE)
        draw.line((x, y, x, y + header_h), fill=WHITE, width=1)
        x += width
    draw.line((x, y, x, y + header_h), fill=WHITE, width=1)
    draw.text((x0, y - 30), "REPEATED_HEADER", font=font(18, "mono"), fill=RED)
    y += header_h
    for row_index, values in enumerate(rows):
        row_fill = WHITE if row_index % 2 == 0 else LIGHT_GRAY
        draw.rectangle((x0, y, x0 + sum(widths), y + row_h), fill=row_fill, outline=(116, 127, 139), width=2)
        x = x0
        for value, width in zip(values, widths):
            draw_wrapped(draw, (x + 10, y + 22), value, font(21), width - 20, line_gap=5, max_lines=2)
            draw.line((x, y, x, y + row_h), fill=(116, 127, 139), width=1)
            x += width
        draw.line((x, y, x, y + row_h), fill=(116, 127, 139), width=1)
        y += row_h
    draw.rounded_rectangle((80, 1500, 1110, 1582), radius=12, fill=(255, 246, 221), outline=AMBER, width=2)
    draw.text(
        (100, 1525),
        f"BOUNDARY_MARKER logical_page={logical_page}; repeated_header=true; next_row_continues=true",
        font=font(19, "mono"),
        fill=INK,
    )


def cross_page_rows(logical_page: int, count: int = 8) -> list[tuple[str, ...]]:
    statuses = ("queued", "review", "approved")
    return [
        (
            f"RID-{logical_page:02d}-{index:02d}",
            f"B{logical_page:02d}",
            f"Synthetic component {logical_page}-{index}",
            str((logical_page + index) % 17 + 1),
            f"{(logical_page * 3 + index * 7) % 90 + 10}.00",
            statuses[(logical_page + index) % len(statuses)],
        )
        for index in range(1, count + 1)
    ]


def cross_page_table_pages(logical_pages: Iterable[int] = range(62, 67)) -> list[Image.Image]:
    images: list[Image.Image] = []
    for logical_page in logical_pages:
        image = page()
        draw_cross_page_table(image, logical_page, cross_page_rows(logical_page))
        images.append(image)
    return images


def draw_table_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    *,
    bold: bool = False,
    centered: bool = False,
    fill: tuple[int, int, int] = INK,
) -> None:
    x0, y0, x1, y1 = box
    text_font = font(22, "bold" if bold else "sans")
    lines = wrap_text(draw, value, text_font, x1 - x0 - 18)
    total_h = len(lines) * (text_font.size + 6) - 6
    y = y0 + max(8, (y1 - y0 - total_h) // 2)
    for line in lines:
        if centered:
            x = x0 + (x1 - x0 - text_width(draw, line, text_font)) / 2
        else:
            x = x0 + 9
        draw.text((x, y), line, font=text_font, fill=fill)
        y += text_font.size + 6


def complex_table_pages() -> list[Image.Image]:
    borderless = page()
    draw, top = draw_page_heading(
        borderless,
        "Borderless table",
        "Whitespace alignment, not ruling lines, defines the six data columns.",
        page_label="TBL-P1",
    )
    x_positions = [88, 250, 430, 690, 845, 1000]
    headers = ["Code", "Region", "Product", "Q1", "Q2", "Delta"]
    for x, value in zip(x_positions, headers):
        draw.text((x, top + 30), value, font=font(24, "bold"), fill=BLUE)
    draw.line((80, top + 73, 1110, top + 73), fill=(125, 137, 150), width=3)
    data = [
        ("A-101", "North", "Nimbus", "120", "138", "+18"),
        ("A-102", "North", "Cirrus", "95", "91", "-4"),
        ("B-201", "West", "Stratus", "210", "246", "+36"),
        ("B-202", "West", "Cumulus", "178", "165", "-13"),
        ("C-301", "East", "Aurora", "81", "116", "+35"),
        ("C-302", "East", "Zephyr", "149", "152", "+3"),
    ]
    y = top + 115
    for row_index, row in enumerate(data):
        if row_index % 2:
            draw.rectangle((80, y - 14, 1110, y + 58), fill=(248, 250, 252))
        for x, value in zip(x_positions, row):
            draw.text((x, y), value, font=font(23), fill=INK)
        y += 116
    draw.text((80, 1110), "No vertical or horizontal cell borders appear in the data body.", font=font(23), fill=RED)

    merged = page()
    draw, top = draw_page_heading(
        merged,
        "Multi-level header and merged cells",
        "The header has column spans; the Region cells have row spans.",
        page_label="TBL-P2",
    )
    x = [80, 300, 545, 720, 895, 1110]
    y0 = top + 30
    h1, h2, rh = 92, 84, 132
    # Header row 1: Region and Product span two rows; Measurements spans three columns.
    cells = [
        ((x[0], y0, x[1], y0 + h1 + h2), "Region", LIGHT_BLUE),
        ((x[1], y0, x[2], y0 + h1 + h2), "Product", LIGHT_BLUE),
        ((x[2], y0, x[5], y0 + h1), "Measurements", (212, 230, 247)),
        ((x[2], y0 + h1, x[3], y0 + h1 + h2), "Mean", LIGHT_GRAY),
        ((x[3], y0 + h1, x[4], y0 + h1 + h2), "Std dev", LIGHT_GRAY),
        ((x[4], y0 + h1, x[5], y0 + h1 + h2), "N", LIGHT_GRAY),
    ]
    for box, value, fill_color in cells:
        draw.rectangle(box, fill=fill_color, outline=INK, width=3)
        draw_table_text(draw, box, value, bold=True, centered=True)
    body_top = y0 + h1 + h2
    rows = [
        ("North", "Nimbus", "42.1", "3.8", "120"),
        (None, "Cirrus", "38.6", "4.1", "95"),
        ("West", "Stratus", "51.4", "2.9", "210"),
        (None, "Cumulus", "49.7", "3.2", "178"),
        ("East", "Aurora", "45.3", "5.0", "81"),
        (None, "Zephyr", "47.9", "3.7", "149"),
    ]
    for row_index, row in enumerate(rows):
        row_y = body_top + row_index * rh
        if row_index % 2 == 0:
            region_box = (x[0], row_y, x[1], row_y + 2 * rh)
            draw.rectangle(region_box, fill=(244, 248, 252), outline=INK, width=3)
            draw_table_text(draw, region_box, row[0] or "", bold=True, centered=True)
        for column_index, value in enumerate(row[1:], start=1):
            box = (x[column_index], row_y, x[column_index + 1], row_y + rh)
            draw.rectangle(box, fill=WHITE, outline=INK, width=2)
            draw_table_text(draw, box, value or "", centered=column_index >= 2)
    return [borderless, merged]


def degraded_image_panel(
    label: str,
    target_text: str,
    *,
    blur_radius: float = 0.0,
    low_contrast: bool = False,
    resolution: tuple[int, int] = (850, 330),
) -> Image.Image:
    panel = Image.new("RGB", resolution, (244, 246, 247) if low_contrast else WHITE)
    draw = ImageDraw.Draw(panel)
    outline = (203, 207, 210) if low_contrast else (60, 69, 80)
    text_color = (190, 194, 197) if low_contrast else (22, 28, 36)
    draw.rectangle((10, 10, resolution[0] - 10, resolution[1] - 10), outline=outline, width=4)
    draw.text((35, 34), label, font=font(30, "bold"), fill=text_color)
    draw.text((35, 112), target_text, font=font(43, "mono"), fill=text_color)
    draw.text((35, 204), "checksum: 7K9-M2Q", font=font(31, "mono"), fill=text_color)
    if blur_radius:
        panel = panel.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return panel


def degraded_image_pages() -> list[Image.Image]:
    image = page()
    draw, top = draw_page_heading(
        image,
        "Blurred and low-contrast image regions",
        "Each target is raster-only; ground truth records the legible source phrase.",
        page_label="IMG-P1",
    )
    sharp = degraded_image_panel("CONTROL", "TARGET: SHARP-ORBIT")
    blurred = degraded_image_panel("GAUSSIAN BLUR r=3.2", "TARGET: BLUR-COMET", blur_radius=3.2)
    contrast = degraded_image_panel("LOW CONTRAST", "TARGET: PALE-MOON", low_contrast=True)
    panels = [(sharp, top + 35), (blurred, top + 430), (contrast, top + 825)]
    for panel_image, y in panels:
        scaled = panel_image.resize((1000, 388), Image.Resampling.LANCZOS)
        image.paste(scaled, (95, y))
    draw.text((95, 1455), "Evaluator should distinguish genuine degradation from missing or occluded content.", font=font(21), fill=RED)
    return [image]


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int] = BLUE) -> None:
    draw.line((*start, *end), fill=color, width=7)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 24
    for delta in (2.55, -2.55):
        tip = (
            end[0] + int(math.cos(angle + delta) * length),
            end[1] + int(math.sin(angle + delta) * length),
        )
        draw.line((*end, *tip), fill=color, width=7)


def node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], token: str, color: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=24, fill=color, outline=INK, width=3)
    text_font = font(25, "bold")
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - text_width(draw, token, text_font)) / 2
    draw.text((x, y0 + (y1 - y0 - text_font.size) / 2 - 4), token, font=text_font, fill=INK)


def diagram_chart_pages() -> list[Image.Image]:
    flow = page()
    draw, top = draw_page_heading(
        flow,
        "Decision flowchart",
        "Extract node labels, directed edges, branch labels, and terminal states.",
        page_label="DGM-P1",
    )
    node(draw, (405, top + 20, 785, top + 135), "START: RECEIVE", (218, 239, 226))
    node(draw, (405, top + 235, 785, top + 370), "VALID FORMAT?", (255, 239, 192))
    node(draw, (120, top + 500, 455, top + 620), "REJECT", (249, 213, 213))
    node(draw, (735, top + 500, 1070, top + 620), "SCORE >= 0.8?", (255, 239, 192))
    node(draw, (585, top + 790, 875, top + 910), "MANUAL REVIEW", (229, 225, 245))
    node(draw, (875, top + 790, 1110, top + 910), "ACCEPT", (218, 239, 226))
    arrow(draw, (595, top + 135), (595, top + 235))
    arrow(draw, (405, top + 302), (285, top + 500))
    arrow(draw, (785, top + 302), (902, top + 500))
    arrow(draw, (820, top + 620), (730, top + 790))
    arrow(draw, (982, top + 620), (993, top + 790))
    draw.text((320, top + 385), "NO", font=font(24, "bold"), fill=RED)
    draw.text((820, top + 385), "YES", font=font(24, "bold"), fill=GREEN)
    draw.text((745, top + 675), "NO", font=font(24, "bold"), fill=RED)
    draw.text((995, top + 675), "YES", font=font(24, "bold"), fill=GREEN)

    chart = page()
    draw, top = draw_page_heading(
        chart,
        "Statistical chart",
        "Grouped bars and a line share categorical x labels but use separate y axes.",
        page_label="DGM-P2",
    )
    plot = (135, top + 80, 1040, 1280)
    x0, y0, x1, y1 = plot
    draw.line((x0, y1, x1, y1), fill=INK, width=4)
    draw.line((x0, y0, x0, y1), fill=INK, width=4)
    draw.line((x1, y0, x1, y1), fill=RED, width=3)
    for tick in range(0, 101, 20):
        y = y1 - int((tick / 100) * (y1 - y0))
        draw.line((x0, y, x1, y), fill=(220, 225, 230), width=2)
        draw.text((78, y - 13), str(tick), font=font(19), fill=MUTED)
    categories = ["Q1", "Q2", "Q3", "Q4"]
    alpha = [42, 58, 73, 66]
    beta = [35, 64, 61, 82]
    rate = [62, 71, 69, 88]
    step = (x1 - x0) // len(categories)
    centers: list[int] = []
    for index, category in enumerate(categories):
        center = x0 + step * index + step // 2
        centers.append(center)
        for offset, value, color in ((-44, alpha[index], BLUE), (8, beta[index], GREEN)):
            bar_h = int(value / 100 * (y1 - y0))
            draw.rectangle((center + offset, y1 - bar_h, center + offset + 36, y1), fill=color)
            draw.text((center + offset - 1, y1 - bar_h - 32), str(value), font=font(18, "bold"), fill=color)
        draw.text((center - 24, y1 + 20), category, font=font(23, "bold"), fill=INK)
    points = []
    for center, value in zip(centers, rate):
        y = y1 - int(value / 100 * (y1 - y0))
        points.append((center, y))
    draw.line(points, fill=RED, width=7)
    for center, (x, y), value in zip(centers, points, rate):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=WHITE, outline=RED, width=5)
        draw.text((center + 18, y - 25), f"{value}%", font=font(18, "bold"), fill=RED)
    draw.rectangle((250, 1390, 940, 1485), fill=LIGHT_GRAY)
    draw.rectangle((280, 1420, 312, 1452), fill=BLUE)
    draw.text((325, 1419), "Alpha", font=font(21), fill=INK)
    draw.rectangle((465, 1420, 497, 1452), fill=GREEN)
    draw.text((510, 1419), "Beta", font=font(21), fill=INK)
    draw.line((660, 1436, 710, 1436), fill=RED, width=6)
    draw.text((725, 1419), "Success rate", font=font(21), fill=INK)
    return [flow, chart]


def fraction(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top_y: int,
    numerator: str,
    denominator: str,
    text_font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    num_w = text_width(draw, numerator, text_font)
    den_w = text_width(draw, denominator, text_font)
    width = int(max(num_w, den_w) + 26)
    x0 = center_x - width // 2
    draw.text((center_x - num_w / 2, top_y), numerator, font=text_font, fill=INK)
    line_y = top_y + text_font.size + 11
    draw.line((x0, line_y, x0 + width, line_y), fill=INK, width=3)
    draw.text((center_x - den_w / 2, line_y + 10), denominator, font=text_font, fill=INK)
    return (x0, top_y, x0 + width, line_y + 10 + text_font.size)


def formula_pages() -> list[Image.Image]:
    first = page()
    draw, top = draw_page_heading(
        first,
        "Complex mathematical formulas",
        "Display math mixes integrals, limits, radicals, matrices, sums, and Greek symbols.",
        page_label="MATH-P1",
    )
    serif = font(43, "serif")
    draw.rounded_rectangle((75, top + 25, 1115, top + 340), radius=16, fill=(251, 250, 246), outline=(176, 169, 145), width=2)
    draw.text((105, top + 60), "Eq. 1", font=font(22, "bold"), fill=BLUE)
    draw.text((210, top + 90), "F(k) = ∫", font=serif, fill=INK)
    draw.text((420, top + 55), "+∞", font=font(25, "serif"), fill=INK)
    draw.text((420, top + 145), "−∞", font=font(25, "serif"), fill=INK)
    draw.text((480, top + 90), "e", font=serif, fill=INK)
    draw.text((514, top + 70), "−i2πkx", font=font(25, "serif"), fill=INK)
    draw.text((645, top + 90), "f(x) dx", font=serif, fill=INK)
    draw.text((210, top + 225), "with  k ∈ ℝ  and  i² = −1", font=font(34, "serif"), fill=INK)

    draw.rounded_rectangle((75, top + 390, 1115, top + 770), radius=16, fill=(247, 250, 253), outline=(145, 169, 195), width=2)
    draw.text((105, top + 425), "Eq. 2", font=font(22, "bold"), fill=BLUE)
    draw.text((190, top + 500), "lim", font=serif, fill=INK)
    draw.text((178, top + 555), "n→∞", font=font(24, "serif"), fill=INK)
    draw.text((285, top + 500), "Σ", font=font(68, "serif"), fill=INK)
    draw.text((306, top + 475), "n", font=font(23, "serif"), fill=INK)
    draw.text((285, top + 580), "j=1", font=font(22, "serif"), fill=INK)
    fraction(draw, 430, top + 480, "(−1)ʲ · j²", "n³ + j", font(31, "serif"))
    draw.text((560, top + 510), "=  −", font=serif, fill=INK)
    fraction(draw, 735, top + 480, "1", "12", font(35, "serif"))
    draw.text((105, top + 680), "Normalized: lim_{n→∞} Σ_{j=1}^{n} ((−1)^j j^2)/(n^3+j) = −1/12", font=font(20, "mono"), fill=MUTED)

    second = page()
    draw, top = draw_page_heading(
        second,
        "Matrices and partial differential equation",
        "Exact normalized strings are stored in ground_truth.json.",
        page_label="MATH-P2",
    )
    draw.rounded_rectangle((75, top + 25, 1115, top + 435), radius=16, fill=(250, 248, 252), outline=(165, 146, 177), width=2)
    draw.text((105, top + 60), "Eq. 3", font=font(22, "bold"), fill=BLUE)
    draw.text((155, top + 160), "A =", font=font(44, "serif"), fill=INK)
    draw.line((275, top + 125, 275, top + 350), fill=INK, width=4)
    draw.line((275, top + 125, 300, top + 125), fill=INK, width=4)
    draw.line((275, top + 350, 300, top + 350), fill=INK, width=4)
    draw.line((690, top + 125, 690, top + 350), fill=INK, width=4)
    draw.line((665, top + 125, 690, top + 125), fill=INK, width=4)
    draw.line((665, top + 350, 690, top + 350), fill=INK, width=4)
    matrix = (("α", "β²", "√γ"), ("0", "eⁱᶿ", "−λ"), ("∂x", "∂y", "1/2"))
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            draw.text((325 + column_index * 120, top + 135 + row_index * 70), value, font=font(34, "serif"), fill=INK)
    draw.text((735, top + 200), "det(A) ≠ 0", font=font(38, "serif"), fill=INK)

    draw.rounded_rectangle((75, top + 490, 1115, top + 890), radius=16, fill=(247, 251, 248), outline=(142, 176, 151), width=2)
    draw.text((105, top + 525), "Eq. 4", font=font(22, "bold"), fill=BLUE)
    draw.text((165, top + 625), "∂u", font=font(39, "serif"), fill=INK)
    draw.line((155, top + 675, 235, top + 675), fill=INK, width=3)
    draw.text((165, top + 690), "∂t", font=font(39, "serif"), fill=INK)
    draw.text((260, top + 665), "+  (u · ∇)u  =  −∇p  +  ν∇²u", font=font(42, "serif"), fill=INK)
    draw.text((205, top + 790), "subject to  ∇ · u = 0", font=font(36, "serif"), fill=INK)
    return [first, second]


def header_footer_watermark_pages() -> list[Image.Image]:
    images: list[Image.Image] = []
    for page_number in range(1, 4):
        image = page()
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        watermark_font = font(94, "bold")
        watermark_text = f"SYNTHETIC COPY {page_number}"
        watermark_w = text_width(overlay_draw, watermark_text, watermark_font)
        overlay_draw.text(((image.width - watermark_w) / 2, 730), watermark_text, font=watermark_font, fill=(100, 115, 130, 34))
        overlay = overlay.rotate(32, center=(image.width // 2, image.height // 2), resample=Image.Resampling.BICUBIC)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 94), fill=(32, 57, 88))
        draw.text((70, 27), "NORTHWIND LABS — SYNTHETIC QUARTERLY BRIEF", font=font(23, "bold"), fill=WHITE)
        right = f"HEADER-P{page_number}"
        draw.text((image.width - 70 - text_width(draw, right, font(20, "mono")), 30), right, font=font(20, "mono"), fill=(210, 226, 243))
        draw.line((70, 1510, image.width - 70, 1510), fill=(141, 151, 162), width=2)
        draw.text((70, 1538), "CONFIDENTIAL — FIXTURE ONLY", font=font(19, "bold"), fill=RED)
        footer = f"FOOTER-P{page_number} | Page {page_number} of 3 | DOC-ID HFW-2026"
        draw.text((image.width - 70 - text_width(draw, footer, font(18, "mono")), 1540), footer, font=font(18, "mono"), fill=MUTED)
        draw.text((80, 170), f"Body section {page_number}", font=font(42, "bold"), fill=INK)
        body = (
            f"BODY-P{page_number}: This paragraph is unique to page {page_number}. Repeated running headers, "
            "footers, page numbers, and diagonal watermarks should be classified separately from body text."
        )
        draw_wrapped(draw, (82, 250), body, font(30), 1010, line_gap=17)
        draw.rounded_rectangle((82, 535, 1108, 880), radius=16, fill=(249, 250, 252), outline=(174, 183, 194), width=2)
        draw.text((115, 580), f"Unique fact {page_number}", font=font(29, "bold"), fill=BLUE)
        draw_wrapped(
            draw,
            (115, 650),
            f"The synthetic index for page {page_number} is {page_number * 137}. Do not deduplicate this body fact.",
            font(28),
            900,
            line_gap=14,
        )
        images.append(image)
    return images


def occlusion_pages() -> list[Image.Image]:
    image = page()
    draw, top = draw_page_heading(
        image,
        "Occluded and irrecoverable fields",
        "Black regions contain no hidden text pixels or PDF text objects. Never infer their values.",
        page_label="OCC-P1",
    )
    fields = [
        ("Record ID", "REC-1042", False),
        ("Contact name", "Ada Example", False),
        ("Access code", "", True),
        ("Account suffix", "", True),
        ("Review status", "APPROVED", False),
    ]
    y = top + 35
    for label, value, hidden in fields:
        draw.rounded_rectangle((105, y, 1085, y + 168), radius=14, fill=(248, 250, 252), outline=(167, 177, 188), width=2)
        draw.text((140, y + 25), label, font=font(23, "bold"), fill=MUTED)
        if hidden:
            # Intentionally draw no source value before covering the field.  This
            # guarantees that neither pixels nor PDF objects contain recoverable data.
            draw.rectangle((430, y + 31, 1015, y + 137), fill=(4, 4, 4))
            marker = "[IRRECOVERABLE]"
            marker_font = font(28, "mono")
            marker_x = 430 + (585 - text_width(draw, marker, marker_font)) / 2
            draw.text((marker_x, y + 66), marker, font=marker_font, fill=(230, 230, 230))
        else:
            draw.text((430, y + 52), value, font=font(32, "mono"), fill=INK)
        y += 225
    draw.rounded_rectangle((105, 1370, 1085, 1510), radius=14, fill=(255, 240, 240), outline=RED, width=2)
    draw.text((140, 1400), "Expected output for both black fields: [OCCLUDED]", font=font(25, "bold"), fill=RED)
    draw.text((140, 1450), "Any guessed value is an evaluation failure.", font=font(23), fill=INK)
    return [image]


@dataclass(frozen=True)
class Fixture:
    filename: str
    scenario_ids: tuple[str, ...]
    build: Callable[[], list[Image.Image]]


FIXTURES: tuple[Fixture, ...] = (
    Fixture("two_column_reading_order.pdf", ("two_column_reading_order",), two_column_pages),
    Fixture("mixed_orientation.pdf", ("horizontal_vertical_text", "mixed_page_orientation"), mixed_orientation_pages),
    Fixture("cross_page_table_boundary_surrogate.pdf", ("cross_page_table", "repeated_table_header", "logical_page_64_boundary"), cross_page_table_pages),
    Fixture("borderless_multilevel_merged_tables.pdf", ("borderless_table", "multilevel_header", "merged_cells"), complex_table_pages),
    Fixture("degraded_images.pdf", ("blurred_image", "low_contrast_image"), degraded_image_pages),
    Fixture("flowchart_statistical_chart.pdf", ("flowchart", "statistical_chart"), diagram_chart_pages),
    Fixture("complex_formulas.pdf", ("complex_formula", "matrix", "partial_differential_equation"), formula_pages),
    Fixture("headers_footers_watermarks.pdf", ("running_header", "running_footer", "page_number", "watermark"), header_footer_watermark_pages),
    Fixture("occlusion_irrecoverable.pdf", ("occlusion", "irrecoverable_field", "anti_hallucination"), occlusion_pages),
)


def jpeg_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(
        output,
        format="JPEG",
        quality=92,
        subsampling=0,
        optimize=False,
        progressive=False,
        dpi=(144, 144),
    )
    return output.getvalue()


def pdf_stream(dictionary: str, data: bytes) -> bytes:
    return f"<< {dictionary} /Length {len(data)} >>\nstream\n".encode("ascii") + data + b"\nendstream"


def write_deterministic_pdf(path: Path, images: Sequence[Image.Image]) -> None:
    """Write image-only PDF bytes with stable metadata and object ordering."""
    if not images:
        raise ValueError("A PDF must contain at least one page")
    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    page_ids = [3 + index * 3 for index in range(len(images))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(images)} /Kids [{kids}] >>".encode("ascii")

    for index, image in enumerate(images):
        page_id = page_ids[index]
        content_id = page_id + 1
        image_id = page_id + 2
        width_px, height_px = image.size
        width_pt = width_px / 2
        height_pt = height_px / 2
        width_text = f"{width_pt:g}"
        height_text = f"{height_pt:g}"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_text} {height_text}] "
            f"/Resources << /XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        content = f"q\n{width_text} 0 0 {height_text} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
        objects[content_id] = pdf_stream("", content)
        jpg = jpeg_bytes(image)
        objects[image_id] = pdf_stream(
            f"/Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} "
            "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode",
            jpg,
        )

    output = io.BytesIO()
    output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(output.tell())
        output.write(f"{object_id} 0 obj\n".encode("ascii"))
        output.write(objects[object_id])
        output.write(b"\nendobj\n")
    xref_offset = output.tell()
    output.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    stable_id = hashlib.md5(path.name.encode("utf-8"), usedforsecurity=False).hexdigest()
    output.write(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R "
            f"/ID [<{stable_id}><{stable_id}>] >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(output.getvalue())


def ground_truth() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "fixture_kind": "fully_synthetic_no_real_user_data",
        "documents": [
            {
                "file": "two_column_reading_order.pdf",
                "page_count": 1,
                "scenario_ids": ["two_column_reading_order"],
                "expected": {
                    "reading_order_tokens": ["DOC_TITLE", "L1", "L2", "L3", "R1", "R2", "R3", "CAPTION"],
                    "column_count": 2,
                    "do_not_interleave_columns": True,
                },
            },
            {
                "file": "mixed_orientation.pdf",
                "page_count": 2,
                "scenario_ids": ["horizontal_vertical_text", "mixed_page_orientation"],
                "expected": {
                    "page_orientations": ["portrait", "landscape"],
                    "horizontal_tokens": ["H1", "H2", "H3", "LANDSCAPE-H1", "LANDSCAPE-H2"],
                    "vertical_token": "V1",
                    "vertical_text_top_to_bottom": "纵向文字测试区",
                    "rotated_text": "ROTATED-90-DEGREES",
                },
            },
            {
                "file": "cross_page_table_boundary_surrogate.pdf",
                "page_count": 5,
                "scenario_ids": ["cross_page_table", "repeated_table_header", "logical_page_64_boundary"],
                "expected": {
                    "physical_to_logical_page": {"1": 62, "2": 63, "3": 64, "4": 65, "5": 66},
                    "logical_boundary": 64,
                    "header": list(TABLE_HEADERS),
                    "header_repeated_on_every_page": True,
                    "rows_per_page": 8,
                    "row_ids_by_logical_page": {
                        str(logical_page): [row[0] for row in cross_page_rows(logical_page)]
                        for logical_page in range(62, 67)
                    },
                    "surrogate_note": "Five physical pages stand in for logical pages 62-66. Physical page 3 is the logical page-64 boundary; this exercises boundary arithmetic without a large default artifact.",
                },
            },
            {
                "file": "borderless_multilevel_merged_tables.pdf",
                "page_count": 2,
                "scenario_ids": ["borderless_table", "multilevel_header", "merged_cells"],
                "expected": {
                    "page_1": {
                        "table_kind": "borderless",
                        "header": ["Code", "Region", "Product", "Q1", "Q2", "Delta"],
                        "rows": [
                            ["A-101", "North", "Nimbus", "120", "138", "+18"],
                            ["A-102", "North", "Cirrus", "95", "91", "-4"],
                            ["B-201", "West", "Stratus", "210", "246", "+36"],
                            ["B-202", "West", "Cumulus", "178", "165", "-13"],
                            ["C-301", "East", "Aurora", "81", "116", "+35"],
                            ["C-302", "East", "Zephyr", "149", "152", "+3"],
                        ],
                    },
                    "page_2": {
                        "header_spans": [
                            {"text": "Region", "rowspan": 2, "colspan": 1},
                            {"text": "Product", "rowspan": 2, "colspan": 1},
                            {"text": "Measurements", "rowspan": 1, "colspan": 3},
                        ],
                        "leaf_headers": ["Mean", "Std dev", "N"],
                        "body_row_spans": [
                            {"text": "North", "rowspan": 2},
                            {"text": "West", "rowspan": 2},
                            {"text": "East", "rowspan": 2},
                        ],
                    },
                },
            },
            {
                "file": "degraded_images.pdf",
                "page_count": 1,
                "scenario_ids": ["blurred_image", "low_contrast_image"],
                "expected": {
                    "control": "TARGET: SHARP-ORBIT",
                    "blurred": "TARGET: BLUR-COMET",
                    "low_contrast": "TARGET: PALE-MOON",
                    "common_checksum": "7K9-M2Q",
                },
            },
            {
                "file": "flowchart_statistical_chart.pdf",
                "page_count": 2,
                "scenario_ids": ["flowchart", "statistical_chart"],
                "expected": {
                    "page_1": {
                        "nodes": ["START: RECEIVE", "VALID FORMAT?", "REJECT", "SCORE >= 0.8?", "MANUAL REVIEW", "ACCEPT"],
                        "edges": [
                            ["START: RECEIVE", "VALID FORMAT?", None],
                            ["VALID FORMAT?", "REJECT", "NO"],
                            ["VALID FORMAT?", "SCORE >= 0.8?", "YES"],
                            ["SCORE >= 0.8?", "MANUAL REVIEW", "NO"],
                            ["SCORE >= 0.8?", "ACCEPT", "YES"],
                        ],
                    },
                    "page_2": {
                        "categories": ["Q1", "Q2", "Q3", "Q4"],
                        "series": {"Alpha": [42, 58, 73, 66], "Beta": [35, 64, 61, 82], "Success rate": [62, 71, 69, 88]},
                    },
                },
            },
            {
                "file": "complex_formulas.pdf",
                "page_count": 2,
                "scenario_ids": ["complex_formula", "matrix", "partial_differential_equation"],
                "expected": {
                    "normalized_latex": [
                        "F(k) = \\int_{-\\infty}^{+\\infty} e^{-i2\\pi kx} f(x) \\, dx",
                        "\\lim_{n\\to\\infty} \\sum_{j=1}^{n} \\frac{(-1)^j j^2}{n^3+j} = -\\frac{1}{12}",
                        "A = \\begin{bmatrix} \\alpha & \\beta^2 & \\sqrt{\\gamma} \\\\ 0 & e^{i\\theta} & -\\lambda \\\\ \\partial x & \\partial y & 1/2 \\end{bmatrix}",
                        "\\frac{\\partial u}{\\partial t} + (u \\cdot \\nabla)u = -\\nabla p + \\nu \\nabla^2 u, \\quad \\nabla \\cdot u = 0",
                    ]
                },
            },
            {
                "file": "headers_footers_watermarks.pdf",
                "page_count": 3,
                "scenario_ids": ["running_header", "running_footer", "page_number", "watermark"],
                "expected": {
                    "page_tokens": [
                        {"page": 1, "header": "HEADER-P1", "body": "BODY-P1", "watermark": "SYNTHETIC COPY 1", "footer": "FOOTER-P1"},
                        {"page": 2, "header": "HEADER-P2", "body": "BODY-P2", "watermark": "SYNTHETIC COPY 2", "footer": "FOOTER-P2"},
                        {"page": 3, "header": "HEADER-P3", "body": "BODY-P3", "watermark": "SYNTHETIC COPY 3", "footer": "FOOTER-P3"},
                    ],
                    "classify_as_non_body": ["running header", "running footer", "page number", "watermark"],
                    "do_not_deduplicate": ["BODY-P1", "BODY-P2", "BODY-P3"],
                },
            },
            {
                "file": "occlusion_irrecoverable.pdf",
                "page_count": 1,
                "scenario_ids": ["occlusion", "irrecoverable_field", "anti_hallucination"],
                "expected": {
                    "visible_fields": {"Record ID": "REC-1042", "Contact name": "Ada Example", "Review status": "APPROVED"},
                    "occluded_fields": {
                        "Access code": {"expected": "[OCCLUDED]", "recoverable": False},
                        "Account suffix": {"expected": "[OCCLUDED]", "recoverable": False},
                    },
                    "forbidden_behavior": "Do not guess, reconstruct, or hallucinate values for blacked-out fields.",
                    "construction_guarantee": "No underlying value is drawn before rasterization; no hidden text object exists in the PDF.",
                },
            },
        ],
    }


PAGE_RE = re.compile(rb"/Type\s*/Page(?!s)\b")


def internal_pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-1.4") or not data.rstrip().endswith(b"%%EOF"):
        raise ValueError(f"{path.name}: missing PDF header or EOF marker")
    return len(PAGE_RE.findall(data))


def pdfinfo_page_count(path: Path) -> int | None:
    command = shutil.which("pdfinfo")
    if not command:
        return None
    result = subprocess.run([command, str(path)], check=True, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ValueError(f"{path.name}: pdfinfo did not report a page count")
    return int(match.group(1))


def validate_pdf(path: Path, expected_pages: int) -> dict[str, Any]:
    internal_count = internal_pdf_page_count(path)
    if internal_count != expected_pages:
        raise ValueError(f"{path.name}: internal page count {internal_count}, expected {expected_pages}")
    pdfinfo_count = pdfinfo_page_count(path)
    if pdfinfo_count is not None and pdfinfo_count != expected_pages:
        raise ValueError(f"{path.name}: pdfinfo page count {pdfinfo_count}, expected {expected_pages}")
    data = path.read_bytes()
    return {
        "file": path.name,
        "page_count": expected_pages,
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "validated_by": ["internal_pdf_structure"] + (["pdfinfo"] if pdfinfo_count is not None else []),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(output_dir: Path, include_full_boundary: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_counts: dict[str, int] = {}
    for fixture in FIXTURES:
        images = fixture.build()
        write_deterministic_pdf(output_dir / fixture.filename, images)
        expected_counts[fixture.filename] = len(images)
    if include_full_boundary:
        filename = "cross_page_table_full_65_pages.pdf"
        images = cross_page_table_pages(range(1, 66))
        write_deterministic_pdf(output_dir / filename, images)
        expected_counts[filename] = 65

    truth = ground_truth()
    write_json(output_dir / "ground_truth.json", truth)
    validation = [validate_pdf(output_dir / name, count) for name, count in sorted(expected_counts.items())]
    manifest = {
        "schema_version": "1.0",
        "generator": "generate_fixtures.py",
        "determinism": {
            "network_required": False,
            "random_seed_required": False,
            "pdf_metadata_timestamps": False,
            "rendering": "Pillow raster pages embedded by deterministic in-repo PDF writer",
        },
        "source_data": "fully synthetic; no external, database, or user data",
        "font_paths": FONT_PATHS,
        "full_65_page_boundary_included": include_full_boundary,
        "artifacts": validation,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--include-full-65-page-boundary",
        action="store_true",
        help="Also generate a 65-page cross-page table. The default five-page surrogate is always generated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = generate(args.output_dir.resolve(), args.include_full_65_page_boundary)
    for artifact in manifest["artifacts"]:
        validators = ", ".join(artifact["validated_by"])
        print(
            f"OK {artifact['file']}: {artifact['page_count']} page(s), "
            f"{artifact['byte_size']} bytes, sha256={artifact['sha256'][:16]}…, validators={validators}"
        )
    print(f"Wrote {len(manifest['artifacts'])} PDFs plus manifest.json and ground_truth.json to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
