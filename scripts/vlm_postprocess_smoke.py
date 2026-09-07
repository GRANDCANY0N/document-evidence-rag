#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from mineru_vlm_rag.adapters.siliconflow import SiliconFlowClient, VLMResult
from mineru_vlm_rag.domain.models import BoundingBox
from mineru_vlm_rag.pdf import render_block
from mineru_vlm_rag.settings import load_settings
from mineru_vlm_rag.tables import compare_table_html, create_table_tiles, merge_table_fragments


DOCUMENT_ID = "fa4ec3d5a1a6e4d709d879104a5f08e5ff63f96212263a39ac76871ed91b4bd6"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    output_dir = (args.output_dir or (
        settings.project_root / "outputs" / f"vlm_postprocess_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    old_root = settings.project_root / "outputs" / DOCUMENT_ID
    extracted = old_root / "mineru" / "extracted" / "images"
    chart = extracted / "649ad6379501f737fc29471eecc1dc94e587bb09f08de58ae1e10700930bc6c7.jpg"
    table = extracted / "80f3f2ab1ee85d6346f89f5a7bfc6762d9295c8e35d0cd4112518a4ef1ab7e60.jpg"
    page = old_root / "pages" / "page-0105.png"
    for path in (chart, table, page):
        if not path.is_file():
            raise FileNotFoundError(path)
    title_crop = output_dir / "page-0105-missing-title-date.png"
    render_block(
        page,
        BoundingBox(x0=425, y0=630, x1=600, y1=690),
        1000,
        1000,
        title_crop,
        padding=16,
    )

    client = SiliconFlowClient(
        api_key=settings.siliconflow_api_key,
        base_url=settings.vlm["api_base"],
        vlm_model=settings.vlm_model,
        embedding_model=settings.embedding_model,
        rerank_model=settings.rerank_model,
        temperature=float(settings.vlm["temperature"]),
        max_tokens=int(settings.vlm["max_tokens"]),
        image_detail=str(settings.vlm["image_detail"]),
        max_retries=int(settings.vlm["max_retries"]),
    )
    try:
        ocr = visual = None
        if not args.table_only:
            logging.info("[1/3] OCR missing title/date: %s", title_crop)
            ocr = client.analyze_images(
                [title_crop],
                "ocr_verify",
                context="PDF第105页表格上方漏失区域；只逐字抄录，不解释。",
            )
            logging.info("[2/3] Visual type and facts: %s", chart)
            visual = client.analyze_images(
                [chart],
                "visual_analyze",
                context="MinerU原始类型=chart；验证真实类型，并提取图中事实。",
            )
        logging.info("[3/3] Multi-level table structure: %s", table)
        table_structure = client.analyze_images(
            [table],
            "table_structure",
            context="复杂多层表头完整表；只定位表头和数据区，不抄录全部数据。",
        )
        tiles = create_table_tiles(
            table,
            output_dir / "table_tiles",
            table_structure.structured_data,
            max_tile_height=600,
            overlap=100,
        )
        tile_results = []
        for index, tile in enumerate(tiles, start=1):
            logging.info("[3/3] Multi-level table tile %s/%s: %s", index, len(tiles), tile)
            tile_results.append(
                client.analyze_images(
                    [tile],
                    "table_to_cells",
                    context=(
                        f"复杂多层表头数据分片={index}/{len(tiles)}；严格保留每一位数字和多层表头；"
                        f"结构={json.dumps(table_structure.structured_data, ensure_ascii=False)[:5000]}"
                    ),
                )
            )
        valid_tile_results = [result for result in tile_results if result.html]
        merged_html = merge_table_fragments(
            [result.html for result in valid_tile_results],
            header_rows=[int(result.structured_data.get("header_rows") or 1) for result in valid_tile_results],
        )
        table_result = VLMResult(
            task="table_to_cells",
            readable=bool(merged_html) and all(result.readable for result in tile_results),
            html=merged_html,
            structured_data={"tile_count": len(tiles)},
            uncertainty=[value for result in tile_results for value in result.uncertainty],
        )
    finally:
        client.close()

    checks = {
        "table_readable": table_result.readable,
        "table_has_html": "<table" in table_result.html.lower(),
        "table_was_split": len(tiles) >= 2,
        "all_table_tiles_have_html": all("<table" in result.html.lower() for result in tile_results),
    }
    content_path = next((old_root / "mineru" / "extracted").glob("*_content_list.json"))
    content_items = json.loads(content_path.read_text(encoding="utf-8"))
    mineru_html = next(
        str(item.get("table_body") or "")
        for item in content_items
        if str(item.get("img_path") or "").endswith(table.name)
    )
    table_crosscheck = compare_table_html(mineru_html, merged_html)
    checks["dual_source_agreed"] = bool(table_crosscheck.get("agreed"))
    if ocr is not None and visual is not None:
        visual_type = str(
            visual.structured_data.get("visual_type") or visual.structured_data.get("type") or ""
        ).strip().lower()
        checks.update(
            {
                "ocr_readable": ocr.readable,
                "ocr_contains_title": "金融业资产简表" in (ocr.text + ocr.summary),
                "ocr_contains_date": "2021年12月31日" in (ocr.text + ocr.summary).replace(" ", ""),
                "visual_readable": visual.readable,
                "visual_type_present": bool(visual_type),
            }
        )
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "model": settings.vlm_model,
        "checks": checks,
        "inputs": {
            "ocr_crop": str(title_crop),
            "chart": str(chart),
            "table": str(table),
        },
        "results": {
            "table_structure": table_structure.model_dump(mode="json"),
            "table_tiles": [result.model_dump(mode="json") for result in tile_results],
            "table_to_cells": table_result.model_dump(mode="json"),
            "table_crosscheck": table_crosscheck,
        },
    }
    if ocr is not None and visual is not None:
        payload["results"]["ocr_verify"] = ocr.model_dump(mode="json")
        payload["results"]["visual_analyze"] = visual.model_dump(mode="json")
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks, "result": str(result_path)}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
