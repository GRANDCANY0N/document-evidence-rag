from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import text

from mineru_vlm_rag.chunking import build_chunks
from mineru_vlm_rag.domain.models import (
    Asset, Block, BlockType, BoundingBox, Document, Page, ResolutionStatus,
)
from mineru_vlm_rag.adapters.siliconflow import SiliconFlowClient, VLMResult
from mineru_vlm_rag.normalization import (
    associate_visual_context,
    assign_section_paths,
    mark_cross_page_duplicate_fragments,
    mark_repeated_templates,
    normalize_document_structure,
    parse_mineru_output,
)
from mineru_vlm_rag.normalization.reading_order import rebuild_reading_order
from mineru_vlm_rag.persistence import MilvusRepository, MySQLRepository
from mineru_vlm_rag.pdf import render_block
from mineru_vlm_rag.pipeline.ingest import (
    IngestionPipeline,
    _exception_details,
    _read_mineru_run_metadata,
    _strict_bool,
    _write_mineru_run_metadata,
)
from mineru_vlm_rag.quality import (
    DetectionContext,
    detect_block_issues,
    detect_solid_occlusions,
    image_quality_flags,
)
from mineru_vlm_rag.quality.completeness import SourceLine, _audit_existing_blocks, number_tokens
from mineru_vlm_rag.tables import (
    compare_table_html, continuation_score, create_table_tiles, merge_table_html,
    revert_unsafe_table_merges, table_continuation_evidence, table_header_text, table_records,
)
from mineru_vlm_rag.workflow import WorkflowRecorder


def _block(document_id: str, page_no: int, content: str, bbox: list[float], block_type=BlockType.TEXT) -> Block:
    return Block(
        document_id=document_id,
        page_no=page_no,
        block_type=block_type,
        bbox=BoundingBox.from_sequence(bbox),
        raw_content=content,
        resolved_content=content,
    )


def test_two_column_order_respects_full_width_separators() -> None:
    blocks = [
        _block("doc", 1, "DOC_TITLE", [20, 10, 980, 50], BlockType.TITLE),
        _block("doc", 1, "R1", [560, 100, 930, 130]),
        _block("doc", 1, "L1", [60, 100, 430, 130]),
        _block("doc", 1, "R2", [560, 160, 930, 190]),
        _block("doc", 1, "L2", [60, 160, 430, 190]),
        _block("doc", 1, "CAPTION", [20, 500, 980, 540]),
    ]
    page = Page(document_id="doc", page_no=1, width=1000, height=1400, blocks=blocks)
    rebuild_reading_order(page)
    assert [block.content for block in page.blocks] == ["DOC_TITLE", "L1", "L2", "R1", "R2", "CAPTION"]
    assert all("LAY-001" in block.issue_flags for block in page.blocks[1:5])


def test_year_prefix_is_ordered_before_two_column_continuation() -> None:
    blocks = [
        _block("doc", 1, "二、稳健性评估", [176, 802, 347, 826], BlockType.TITLE),
        _block("doc", 1, "末，资本充足率为15.74%。", [524, 804, 884, 874]),
        _block("doc", 1, "资本充足水平总体稳定。截至2024年", [173, 855, 495, 872]),
    ]
    page = Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=blocks)

    rebuild_reading_order(page)

    assert [block.content for block in page.blocks] == [
        "二、稳健性评估",
        "资本充足水平总体稳定。截至2024年",
        "末，资本充足率为15.74%。",
    ]
    assert page.metadata["reading_order_continuation_repairs"] == 1


def test_heading_spanning_usable_body_separates_two_column_region() -> None:
    title = _block("doc", 1, "专栏三 做好科技金融大文章", [137, 363, 757, 385])
    title.metadata["text_level"] = 2
    intro = _block("doc", 1, "专栏引言。", [137, 411, 472, 662])
    section = _block("doc", 1, "一、科技金融服务能力建设", [136, 671, 470, 714])
    section.metadata["text_level"] = 2
    first = _block("doc", 1, "一是完善政策框架。", [137, 723, 470, 897])
    second = _block("doc", 1, "二是完善金融工具。", [500, 436, 836, 636])
    third = _block("doc", 1, "三是加强市场支持。", [500, 645, 836, 897])
    page = Page(
        document_id="doc", page_no=1, width=1000, height=1000,
        blocks=[title, intro, second, third, section, first],
    )

    rebuild_reading_order(page)

    assert [block.content for block in page.blocks] == [
        title.content,
        intro.content,
        section.content,
        first.content,
        second.content,
        third.content,
    ]


def test_exact_page_top_carry_over_is_preserved_but_not_chunked() -> None:
    previous = _block("doc", 1, "上一页正文结束，大而不能倒风险得到有效缓解。", [100, 700, 900, 890])
    duplicate = _block("doc", 2, "而不能倒风险得到有效缓解。", [100, 110, 400, 140])
    new_text = _block("doc", 2, "下一段新内容。", [100, 180, 900, 240])
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=2,
        pages=[
            Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=[previous]),
            Page(document_id="doc", page_no=2, width=1000, height=1000, blocks=[duplicate, new_text]),
        ],
    )

    count = mark_cross_page_duplicate_fragments(document)
    chunks = build_chunks(document)

    assert count == 1
    assert duplicate.metadata["skip_chunk"] is True
    assert duplicate.metadata["cross_page_duplicate_of"] == previous.block_id
    assert all("而不能倒风险得到有效缓解" not in chunk.display_text for chunk in chunks[1:])
    assert any("下一段新内容" in chunk.display_text for chunk in chunks)


def test_repeated_templates_are_preserved_but_not_chunked() -> None:
    pages = []
    for page_no in range(1, 4):
        pages.append(Page(
            document_id="doc",
            page_no=page_no,
            width=1000,
            height=1000,
            blocks=[
                _block("doc", page_no, "机密文件", [100, 10, 900, 35]),
                _block("doc", page_no, "第一章 范围" if page_no == 1 else f"正文-{page_no}", [100, 150, 900, 210], BlockType.TITLE if page_no == 1 else BlockType.TEXT),
                _block("doc", page_no, f"第{page_no}页", [450, 950, 550, 980]),
            ],
        ))
    document = Document(document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64, page_count=3, pages=pages)
    mark_repeated_templates(document)
    for page in document.pages:
        rebuild_reading_order(page)
    assign_section_paths(document)
    chunks = build_chunks(document)
    assert all(page.blocks[-2].block_type == BlockType.HEADER for page in document.pages)
    assert all(page.blocks[-1].block_type == BlockType.PAGE_NUMBER for page in document.pages)
    assert all("机密文件" not in chunk.display_text and "页" not in chunk.display_text for chunk in chunks)
    assert any(chunk.section_path == "第一章 范围" for chunk in chunks if "正文-2" in chunk.display_text)


def test_visual_context_uses_nearest_heading_and_header_is_not_chunked() -> None:
    heading = _block("doc", 1, "Decision flowchart", [50, 20, 600, 60], BlockType.HEADER)
    visual = _block("doc", 1, "Start -> Validate -> Accept", [50, 100, 900, 700], BlockType.IMAGE)
    document = Document(
        document_id="doc",
        file_name="x.pdf",
        source_path="x.pdf",
        sha256="a" * 64,
        page_count=1,
        pages=[Page(document_id="doc", page_no=1, width=1000, height=1400, blocks=[heading, visual])],
    )
    associate_visual_context(document)
    chunks = build_chunks(document)
    assert len(chunks) == 1
    assert "图题：Decision flowchart" in chunks[0].display_text
    assert chunks[0].block_type == "image"


def test_quality_gate_detects_low_contrast_dark_and_redacted(tmp_path: Path) -> None:
    context = DetectionContext()
    low_contrast = tmp_path / "low.png"
    Image.fromarray(np.full((100, 100), 128, dtype=np.uint8)).save(low_contrast)
    flags = image_quality_flags(low_contrast, context)
    assert {"SRC-002", "SRC-003"} <= set(flags)

    dark = tmp_path / "dark.png"
    Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(dark)
    assert "SRC-004" in image_quality_flags(dark, context)

    redacted = _block("doc", 1, "证件号码：████████", [0, 0, 100, 20])
    assert "SRC-004" in detect_block_issues(redacted, context)

    page = np.full((1000, 1000), 255, dtype=np.uint8)
    page[400:450, 200:800] = 0
    redaction_page = tmp_path / "redaction-page.png"
    Image.fromarray(page).save(redaction_page)
    regions = detect_solid_occlusions(redaction_page)
    assert len(regions) == 1
    assert regions[0].width == 600 and regions[0].height == 50


def test_cross_page_table_score_merge_and_boolean_parsing() -> None:
    first_html = "<table><thead><tr><th>ID</th><th>金额</th></tr></thead><tbody><tr><td>A</td><td>10</td></tr></tbody></table>"
    second_html = "<table><thead><tr><th>ID</th><th>金额</th></tr></thead><tbody><tr><td>B</td><td>20</td></tr></tbody></table>"
    first = _block("doc", 1, first_html, [0, 600, 1000, 990], BlockType.TABLE)
    second = _block("doc", 2, second_html, [0, 10, 1000, 400], BlockType.TABLE)
    assert continuation_score(first, second, 1.0) >= 0.8
    merged = merge_table_html(first_html, second_html, previous_page_no=1, current_page_no=2)
    assert merged.count("金额") == 1
    assert "A" in merged and "B" in merged
    assert 'data-source-page="1"' in merged and 'data-source-page="2"' in merged
    first.resolved_content = merged
    first.metadata["source_pages"] = [1, 2]
    document = Document(
        document_id="doc",
        file_name="x.pdf",
        source_path="x.pdf",
        sha256="a" * 64,
        page_count=2,
        pages=[Page(document_id="doc", page_no=1, blocks=[first])],
    )
    chunks = build_chunks(document)
    assert all(chunk.page_end == 2 for chunk in chunks)
    assert any("来源页：2" in chunk.display_text for chunk in chunks)
    assert _strict_bool(True) and _strict_bool("true") and _strict_bool("是")
    assert not _strict_bool(False) and not _strict_bool("false") and not _strict_bool(None)
    no_thead = "<table><tr><td>ID</td><td>金额</td></tr><tr><td>A</td><td>10</td></tr></table>"
    assert table_header_text(no_thead) == "ID | 金额"
    normalized_merge = merge_table_html(
        no_thead,
        "<table><tr><td>ID</td><td>金额</td></tr><tr><td>B</td><td>20</td></tr></table>",
        previous_page_no=1,
        current_page_no=2,
    )
    assert normalized_merge.count("<thead>") == 1 and normalized_merge.count("<tbody>") == 1
    assert all(f'data-source-page="{page}"' in normalized_merge for page in (1, 2))


def test_page_64_work_units_and_table_boundary_are_not_disconnected(tmp_path: Path) -> None:
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(
        pipeline={"page_work_unit": 64},
        quality={"table_continuation_auto_threshold": 0.78, "table_continuation_vlm_threshold": 0.48},
    )
    pipeline._table_similarity = lambda _left, _right: 1.0
    recorder = WorkflowRecorder("doc64", tmp_path / "workflow64")
    pages = [Page(document_id="doc64", page_no=index) for index in range(1, 66)]
    empty_document = Document(
        document_id="doc64", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=65, pages=pages,
    )
    pipeline._process_blocks(empty_document, Path("x.pdf"), tmp_path, recorder, use_vlm=False)
    completed = [event for event in recorder.events if event.stage == "page_work_unit" and event.decision == "completed"]
    assert [(event.details["page_start"], event.details["page_end"]) for event in completed] == [(1, 64), (65, 65)]

    html1 = "<table><thead><tr><th>ID</th></tr></thead><tbody><tr><td>R64</td></tr></tbody></table>"
    html2 = "<table><thead><tr><th>ID</th></tr></thead><tbody><tr><td>R65</td></tr></tbody></table>"
    html3 = "<table><thead><tr><th>ID</th></tr></thead><tbody><tr><td>R66</td></tr></tbody></table>"
    left = _block("doc64", 64, html1, [0, 600, 1000, 990], BlockType.TABLE)
    right = _block("doc64", 65, html2, [0, 10, 1000, 400], BlockType.TABLE)
    third = _block("doc64", 66, html3, [0, 10, 1000, 400], BlockType.TABLE)
    boundary_document = Document(
        document_id="doc64", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=65,
        pages=[
            Page(document_id="doc64", page_no=64, blocks=[left]),
            Page(document_id="doc64", page_no=65, blocks=[right]),
            Page(document_id="doc64", page_no=66, blocks=[third]),
        ],
    )
    pipeline._merge_cross_page_tables(boundary_document, Path("x.pdf"), tmp_path, recorder, use_vlm=False)
    assert right.metadata["skip_chunk"] is True
    assert third.metadata["skip_chunk"] is True
    assert left.metadata["source_pages"] == [64, 65, 66]
    assert all(value in left.content for value in ("R64", "R65", "R66"))
    assert all(f'data-source-page="{page}"' in left.content for page in (64, 65, 66))


def test_distinct_appendix_titles_block_cross_page_merge_and_revert_old_chain(tmp_path: Path) -> None:
    table_html = (
        "<table><tr><td>项目</td><td>第一季度</td></tr>"
        "<tr><td>资产</td><td>10</td></tr></table>"
    )
    first = _block("doc", 100, table_html, [115, 180, 855, 873], BlockType.TABLE)
    second = _block("doc", 101, table_html.replace("10", "20"), [139, 180, 882, 879], BlockType.TABLE)
    first_ref = _block("doc", 100, "附表7", [147, 120, 205, 137])
    first_title = _block("doc", 100, "2024年存款性公司概览", [394, 120, 578, 137])
    second_ref = _block("doc", 101, "附表8", [174, 120, 226, 137])
    second_title = _block("doc", 101, "2024年货币当局资产负债表", [401, 120, 618, 137])
    for context in (first_ref, first_title):
        context.metadata["associated_table_block_id"] = first.block_id
    for context in (second_ref, second_title):
        context.metadata["associated_table_block_id"] = second.block_id
    pages = [
        Page(document_id="doc", page_no=100, blocks=[first_ref, first_title, first]),
        Page(document_id="doc", page_no=101, blocks=[second_ref, second_title, second]),
    ]
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=101, pages=pages,
    )
    evidence = table_continuation_evidence(first, second, pages[0], pages[1])
    assert evidence["hard_negative"] is True
    assert evidence["distinct_reference"] is True
    assert evidence["auto_eligible"] is False

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(
        embedding={"dimensions": 8},
        quality={"table_continuation_auto_threshold": 0.78, "table_continuation_vlm_threshold": 0.48},
    )
    pipeline._table_similarity = lambda *_args: 1.0
    recorder = WorkflowRecorder("doc", tmp_path / "workflow-distinct")
    pipeline._merge_cross_page_tables(document, Path("x.pdf"), tmp_path, recorder, use_vlm=False)
    assert second.metadata.get("skip_chunk") is not True
    assert first.content == table_html

    # Simulate a database graph written by the old implementation.
    first.resolved_content = merge_table_html(table_html, second.raw_content, 100, 101)
    first.metadata["source_pages"] = [100, 101]
    first.status = ResolutionStatus.REPAIRED
    second.metadata["skip_chunk"] = True
    second.metadata["merged_into"] = first.block_id
    summary = revert_unsafe_table_merges(document)
    assert summary["reverted_parent_count"] == 1
    assert summary["reverted_child_count"] == 1
    assert first.content == first.raw_content
    assert second.metadata.get("skip_chunk") is not True
    assert second.metadata.get("merged_into") is None


def test_table_context_does_not_pollute_sections_and_caption_follows_table() -> None:
    chapter = _block("doc", 1, "第一章 金融市场", [100, 50, 500, 80])
    chapter.metadata["text_level"] = 2
    paragraph = _block("doc", 1, "左栏正文。", [120, 120, 470, 170])
    right = _block("doc", 1, "右栏续文。", [500, 120, 850, 170])
    caption = _block("doc", 1, "未通过压力测试基金数量和占比", [360, 195, 610, 210], BlockType.TITLE)
    table = _block(
        "doc", 1,
        "<table><tr><td>项目</td><td>数量</td></tr><tr><td>A</td><td>1</td></tr></table>",
        [127, 215, 845, 677], BlockType.TABLE,
    )
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=1,
        pages=[Page(document_id="doc", page_no=1, width=1000, height=1400, blocks=[chapter, right, paragraph, caption, table])],
    )
    normalize_document_structure(document)
    ordered = document.pages[0].blocks
    assert [block.content for block in ordered] == [
        "第一章 金融市场", "左栏正文。", "右栏续文。", "未通过压力测试基金数量和占比", table.raw_content,
    ]
    assert caption.metadata["associated_table_block_id"] == table.block_id
    assert table.metadata["caption"] == caption.content
    assert paragraph.section_path == ["第一章 金融市场"]
    assert table.section_path == ["第一章 金融市场"]


def test_no_header_table_keeps_first_data_row_and_td_multilevel_header_is_unique() -> None:
    no_header = (
        "<table><tr><td>情景1</td><td>假设参试银行发生信用违约，考察其可能产生的溢出效应</td></tr>"
        "<tr><td>情景2</td><td>假设非银行金融机构首先发生违约</td></tr></table>"
    )
    headers, records = table_records(no_header)
    assert headers == ["第1列", "第2列"]
    assert len(records) == 2
    assert records[0]["cells"]["第1列"] == "情景1"

    multilevel = (
        "<table><tr><td rowspan='2'>基金类型</td><td colspan='2'>未通过数量(家)</td>"
        "<td colspan='2'>占比(%)</td></tr>"
        "<tr><td>轻度</td><td>重度</td><td>轻度</td><td>重度</td></tr>"
        "<tr><td>股票型</td><td>1</td><td>2</td><td>3%</td><td>4%</td></tr></table>"
    )
    headers, records = table_records(multilevel)
    assert headers == [
        "基金类型", "未通过数量(家) / 轻度", "未通过数量(家) / 重度", "占比(%) / 轻度", "占比(%) / 重度",
    ]
    assert len(records[0]["cells"]) == 5


def test_mineru_normalizer_preserves_source_and_coordinate_space(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    images = extracted / "images"
    images.mkdir(parents=True)
    image_path = images / "table.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    payload = [
        {
            "page_idx": 0,
            "type": "table",
            "bbox": [10, 20, 900, 600],
            "page_size": [1000, 1400],
            "table_body": "<table><tr><td>A</td></tr></table>",
            "img_path": "images/table.png",
        }
    ]
    (extracted / "demo_content_list.json").write_text(json.dumps(payload), encoding="utf-8")
    document = Document(document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64, page_count=1)
    parse_mineru_output(document, extracted)
    block = document.pages[0].blocks[0]
    assert block.block_type == BlockType.TABLE
    assert block.content.startswith("<table>")
    assert document.pages[0].metadata["mineru_coordinate_width"] == 1000
    assert block.asset_id == document.assets[0].asset_id
    assert block.revisions[0].source == "mineru"


def test_mineru_normalizer_infers_api_normalized_coordinate_space(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    payload = [{"page_idx": 0, "type": "text", "bbox": [844, 143, 870, 373], "text": "竖排侧栏"}]
    (extracted / "demo_content_list.json").write_text(json.dumps(payload), encoding="utf-8")
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64, page_count=1
    )

    parse_mineru_output(document, extracted)

    metadata = document.pages[0].metadata
    assert metadata["mineru_coordinate_width"] == 1000
    assert metadata["mineru_coordinate_height"] == 1000
    assert metadata["mineru_coordinate_space"] == "normalized_1000_inferred"


def test_render_block_handles_normalized_reversed_and_outside_boxes(tmp_path: Path) -> None:
    page_image = tmp_path / "page.png"
    Image.new("RGB", (1871, 2521), "white").save(page_image)

    normalized_details: dict = {}
    normalized_crop = tmp_path / "normalized.png"
    render_block(
        page_image,
        BoundingBox.from_sequence([844, 143, 870, 373]),
        1000,
        1000,
        normalized_crop,
        crop_details=normalized_details,
    )
    with Image.open(normalized_crop) as image:
        assert image.width > 2 and image.height > 2
        assert image.size != (1871, 2521)
    assert normalized_details["mode"] == "bbox_crop"

    reversed_details: dict = {}
    reversed_crop = tmp_path / "reversed.png"
    render_block(
        page_image,
        BoundingBox.from_sequence([870, 373, 844, 143]),
        1000,
        1000,
        reversed_crop,
        crop_details=reversed_details,
    )
    assert reversed_details["mode"] == "bbox_crop"

    fallback_details: dict = {}
    fallback_crop = tmp_path / "fallback.png"
    render_block(
        page_image,
        BoundingBox.from_sequence([1200, 100, 1300, 200]),
        1000,
        1000,
        fallback_crop,
        crop_details=fallback_details,
    )
    with Image.open(fallback_crop) as image:
        assert image.size == (1871, 2521)
    assert fallback_details["mode"] == "full_page_fallback"
    assert fallback_details["reason"] == "bbox_outside_or_degenerate"


def test_siliconflow_clients_ignore_proxy_environment_and_disable_sdk_retries() -> None:
    client = SiliconFlowClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        vlm_model="test-vlm",
        embedding_model="test-embedding",
        rerank_model="test-rerank",
    )
    try:
        assert client.openai.max_retries == 0
        assert client.openai._client._trust_env is False
        assert client.http._trust_env is False
    finally:
        client.close()


def test_complex_image_step_records_preparation_result_and_provider_error(tmp_path: Path) -> None:
    image_path = tmp_path / "table.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    table = _block(
        "doc",
        1,
        "<table><tr><td>旧值</td></tr></table>",
        [10, 10, 900, 600],
        BlockType.TABLE,
    )
    table.issue_flags = ["TAB-002"]
    asset = Asset(
        document_id="doc",
        page_no=1,
        kind="table",
        mime_type="image/png",
        sha256="b" * 64,
        source_path=str(image_path),
    )
    table.asset_id = asset.asset_id
    page = Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=[table])
    document = Document(
        document_id="doc",
        file_name="x.pdf",
        source_path="x.pdf",
        sha256="a" * 64,
        page_count=1,
        pages=[page],
        assets=[asset],
    )
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(vlm_model="test-vlm")
    pipeline.detection_context = DetectionContext()
    pipeline.silicon = SimpleNamespace(
        analyze_images=lambda _images, task, context="": VLMResult(
            task=task,
            readable=True,
            html="<table><tr><td>修订值</td></tr></table>",
        )
    )
    recorder = WorkflowRecorder("doc", tmp_path / "workflow")

    pipeline._process_block(document, page, table, Path("x.pdf"), tmp_path, recorder, use_vlm=True)

    assert table.status.value == "review"
    assert table.metadata["chunk_eligible"] is False
    assert "TAB-010" in table.issue_flags
    preparation = next(event for event in recorder.events if event.stage == "image_preparation")
    result = next(event for event in recorder.events if event.stage == "vlm")
    assert preparation.details["image_mode"] == "mineru_asset"
    assert preparation.details["task"] == "table_to_cells"
    assert result.details["duration_seconds"] >= 0

    class ProviderError(Exception):
        status_code = 500
        request_id = "request-123"
        body = {"code": 50507, "message": "Request failed: Unknown error.", "data": None}

    assert _exception_details(ProviderError()) == {
        "error_type": "ProviderError",
        "http_status": 500,
        "request_id": "request-123",
        "provider_code": 50507,
        "provider_message": "Request failed: Unknown error.",
    }


def test_chunk_mysql_milvus_and_workflow_local_roundtrip(tmp_path: Path) -> None:
    table = _block(
        "doc",
        1,
        "<table><thead><tr><th>ID</th><th>值</th></tr></thead><tbody><tr><td>A</td><td>10</td></tr></tbody></table>",
        [0, 0, 100, 100],
        BlockType.TABLE,
    )
    payload = b"asset"
    asset = Asset(document_id="doc", page_no=1, kind="table", mime_type="image/png", sha256="b" * 64, data=payload)
    table.asset_id = asset.asset_id
    document = Document(
        document_id="doc",
        file_name="x.pdf",
        source_path="x.pdf",
        sha256="a" * 64,
        page_count=1,
        pages=[Page(document_id="doc", page_no=1, blocks=[table])],
        assets=[asset],
    )
    chunks = build_chunks(document)
    assert len(chunks) == 2
    assert any("数据行" in chunk.embedding_text for chunk in chunks)

    mysql = MySQLRepository(f"sqlite+pysqlite:///{tmp_path / 'metadata.db'}")
    mysql.initialize()
    mysql.save_document_graph(document, chunks)
    mysql.mark_chunks_embedded(chunk.chunk_id for chunk in chunks)
    mysql.update_document_status("doc", "completed")
    assert mysql.fetch_asset(asset.asset_id) == ("image/png", payload)
    with mysql.engine.connect() as connection:
        assert connection.execute(text("SELECT status FROM documents WHERE document_id='doc'")).scalar() == "completed"
        assert connection.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding_state='embedded'")).scalar() == 2

    milvus = MilvusRepository(str(tmp_path / "milvus.db"), collection_prefix="test_chunks")
    try:
        vectors = [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]]
        milvus.upsert(chunks, vectors)
        hits = milvus.search([1.0, 0.0, 0.0], document_id="doc", limit=2)
        assert hits and hits[0]["document_id"] == "doc"
        replacement = chunks[0].model_copy(update={"chunk_id": "replacement-chunk"})
        milvus.upsert([replacement], [[1.0, 0.0, 0.0]])
        replacement_hits = milvus.search([1.0, 0.0, 0.0], document_id="doc", limit=10)
        assert [hit["id"] for hit in replacement_hits] == ["replacement-chunk"]
        incremental = chunks[1].model_copy(update={"chunk_id": "incremental-chunk"})
        milvus.upsert([incremental], [[0.0, 1.0, 0.0]], replace_documents=False)
        incremental_hits = milvus.search([1.0, 0.0, 0.0], document_id="doc", limit=10)
        assert {hit["id"] for hit in incremental_hits} == {"replacement-chunk", "incremental-chunk"}
    finally:
        milvus.close()

    recorder = WorkflowRecorder("doc", tmp_path / "workflow")
    recorder.record("quality_gate", "detect", "vlm", page_no=1, issue_flags=["TAB-002"], details={"reason": "merged_cells"})
    summary = recorder.write_markdown_summary()
    assert recorder.jsonl_path.exists() and summary.exists()
    assert "TAB-002" in summary.read_text(encoding="utf-8")


def test_multilevel_table_is_canonicalized_and_chunked_with_header_paths() -> None:
    content = (
        "<table><thead>"
        "<tr><th rowspan='2'>项目</th><th colspan='2'>2021年</th></tr>"
        "<tr><th>金额</th><th>同比</th></tr>"
        "</thead><tbody><tr><td>资产</td><td>10.25</td><td>3.1%</td></tr></tbody></table>"
    )
    headers, records = table_records(content)
    assert headers == ["项目", "2021年 / 金额", "2021年 / 同比"]
    assert records[0]["cells"]["2021年 / 金额"] == "10.25"
    assert compare_table_html(content, content)["agreed"] is True
    mineru_format = (
        "<table><tr><th>项目</th><th>值</th></tr>"
        "<tr><td>风险 $^{5}$</td><td>冲击1:上升 $100\\%^{1}$</td></tr></table>"
    )
    vlm_format = (
        "<table><tr><th>项目</th><th>值</th></tr>"
        "<tr><td>风险5</td><td>• 冲击1:上升100%1</td></tr></table>"
    )
    format_comparison = compare_table_html(mineru_format, vlm_format)
    assert format_comparison["agreed"] is True
    assert format_comparison["conflict_count"] == 0
    assert format_comparison["format_difference_count"] == 2
    numeric_conflict = compare_table_html(
        vlm_format,
        vlm_format.replace("100%1", "100.1%1"),
    )
    assert numeric_conflict["agreed"] is False
    assert numeric_conflict["conflict_count"] == 1

    table = _block("doc", 1, content, [10, 10, 990, 900], BlockType.TABLE)
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=1, pages=[Page(document_id="doc", page_no=1, blocks=[table])],
    )
    chunks = build_chunks(document)
    assert any("2021年 / 金额=10.25" in chunk.embedding_text for chunk in chunks)


def test_review_and_unreadable_mineru_blocks_create_provisional_chunks() -> None:
    accepted = _block("doc", 1, "可入库", [0, 0, 100, 20])
    review = _block("doc", 1, "待复核", [0, 30, 100, 50])
    unreadable = _block("doc", 1, "不可读", [0, 60, 100, 80])
    review.status = ResolutionStatus.REVIEW
    unreadable.status = ResolutionStatus.UNREADABLE
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=1,
        pages=[Page(document_id="doc", page_no=1, blocks=[accepted, review, unreadable])],
    )
    chunks = build_chunks(document)
    assert [chunk.display_text for chunk in chunks] == ["可入库", "待复核", "不可读"]
    assert [chunk.metadata["retrieval_tier"] for chunk in chunks] == [
        "verified", "provisional", "provisional",
    ]
    assert chunks[1].metadata["source_content_basis"][review.block_id] == "mineru_raw"


def test_visual_table_uses_structure_tiles_and_dual_source_gate(tmp_path: Path) -> None:
    image_path = tmp_path / "visual-table.png"
    Image.new("RGB", (500, 500), "white").save(image_path)
    visual = _block("doc", 1, "", [0, 0, 1000, 1000], BlockType.IMAGE)
    asset = Asset(
        document_id="doc", page_no=1, kind="image", mime_type="image/png",
        sha256="b" * 64, source_path=str(image_path),
    )
    visual.asset_id = asset.asset_id
    page = Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=[visual])
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=1, pages=[page], assets=[asset],
    )
    html = "<table><thead><tr><th>项目</th><th>值</th></tr></thead><tbody><tr><td>A</td><td>1.25</td></tr></tbody></table>"
    responses = iter([
        VLMResult(
            task="visual_analyze", readable=True,
            structured_data={
                "visual_type": "table",
                "header_region": [0, 0, 1, 0.2],
                "data_region": [0, 0.2, 1, 1],
            },
        ),
        VLMResult(task="table_to_cells", readable=True, html=html),
    ])
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(vlm_model="test-vlm")
    pipeline.detection_context = DetectionContext()
    pipeline.silicon = SimpleNamespace(analyze_images=lambda *_args, **_kwargs: next(responses))
    pipeline._secondary_mineru_table = lambda *_args, **_kwargs: html
    recorder = WorkflowRecorder("doc", tmp_path / "workflow-visual-table")

    pipeline._process_block(document, page, visual, Path("x.pdf"), tmp_path, recorder, use_vlm=True)

    assert visual.block_type == BlockType.TABLE
    assert visual.status == ResolutionStatus.REPAIRED
    assert visual.metadata["chunk_eligible"] is True
    assert "TAB-009" in visual.issue_flags
    assert "dual_source_agreed" in visual.metadata["verification_tags"]


def test_pdf_text_recovery_is_logged_as_accepted_without_vlm(tmp_path: Path) -> None:
    recovered = _block("doc", 1, "金融业资产简表", [100, 100, 400, 140])
    recovered.source = "pdf_text_layer"
    recovered.issue_flags = ["TXT-004"]
    recovered.metadata["verification_tags"] = ["pdf_text_layer_recovered"]
    page = Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=[recovered])
    document = Document(
        document_id="doc", file_name="x.pdf", source_path="x.pdf", sha256="a" * 64,
        page_count=1, pages=[page],
    )
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(vlm_model="test-vlm")
    pipeline.detection_context = DetectionContext()
    pipeline.silicon = SimpleNamespace(
        analyze_images=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("VLM must not run"))
    )
    recorder = WorkflowRecorder("doc", tmp_path / "workflow-pdf-text")

    pipeline._process_block(document, page, recovered, Path("x.pdf"), tmp_path, recorder, use_vlm=True)

    quality = next(event for event in recorder.events if event.stage == "quality_gate")
    assert quality.decision == "accept_pdf_text"
    assert not any(event.stage == "vlm" for event in recorder.events)
    short = _block("doc", 1, "目录", [0, 0, 10, 10])
    assert "TXT-002" not in detect_block_issues(short, DetectionContext())


def test_pdf_text_observation_never_overwrites_nonempty_mineru_text() -> None:
    footnote = _block("doc", 1, "① 课题组成员：杨雨亭 尹明", [100, 100, 500, 140], BlockType.FOOTNOTE)
    page = Page(document_id="doc", page_no=1, width=1000, height=1000, blocks=[footnote])
    findings = _audit_existing_blocks(
        page,
        [
            SourceLine(
                page_no=1,
                line_no=0,
                text="a 课题组成员：杨雨亭 尹明",
                bbox=BoundingBox.from_sequence([100, 100, 500, 140]),
            )
        ],
        text_layer_reliable=True,
    )

    assert footnote.content == "① 课题组成员：杨雨亭 尹明"
    assert footnote.source == "mineru"
    assert footnote.status == ResolutionStatus.ACCEPTED
    assert "TXT-003" in footnote.issue_flags
    assert footnote.metadata["pdf_text_comparison"]["footnote_encoding_conflict"] is True
    assert footnote.metadata["pdf_text_comparison"]["mineru_preserved"] is True
    assert footnote.revisions[-1].source == "pdf_text_layer_observation"
    assert findings[0]["kind"] == "block_text_difference_observed"
    assert number_tokens(r"5\%") == ["5%"]


def test_cross_column_text_layer_truncation_preserves_complete_mineru_text() -> None:
    raw = (
        "质押式回购利率为1.51%，同比下降27个基点；"
        "转贴现利率为1.38%，同比下降21个基点；"
        "质押式回购利率为1.79%，同比下降1个基点。"
    )
    truncated = raw.removesuffix("率为1.79%，同比下降1个基点。")
    block = _block("doc", 56, raw, [110, 578, 473, 883])
    page = Page(document_id="doc", page_no=56, width=1000, height=1000, blocks=[block])
    findings = _audit_existing_blocks(
        page,
        [
            SourceLine(
                page_no=56,
                line_no=55,
                text=truncated,
                bbox=BoundingBox.from_sequence([119, 578, 469, 880]),
            ),
            SourceLine(
                page_no=56,
                line_no=3,
                text="率为1.79%，同比下降1个基点。",
                bbox=BoundingBox.from_sequence([507, 123, 756, 136]),
            ),
        ],
        text_layer_reliable=True,
    )

    assert block.content == raw
    assert block.source == "mineru"
    assert "TXT-003" in block.issue_flags
    assert findings[0]["reference_excerpt"] == truncated
    assert findings[0]["mineru_preserved"] is True


def test_text_vlm_conflict_preserves_mineru_and_blocks_chunking() -> None:
    raw = "质押式回购利率为1.79%，同比下降1个基点。"
    truncated = "质押式回购利"
    block = _block("doc", 56, raw, [110, 578, 473, 883])
    block.issue_flags = ["TXT-003"]
    block.metadata["pdf_text_comparison"] = {"reference_excerpt": truncated}
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(vlm_model="test-vlm")

    pipeline._apply_vlm_result(
        block,
        VLMResult(task="ocr_verify", readable=True, text=truncated),
    )

    assert block.content == raw
    assert block.source == "mineru"
    assert block.status == ResolutionStatus.REVIEW
    assert block.metadata["chunk_eligible"] is False
    assert "vlm_conflict_mineru_preserved" in block.metadata["verification_tags"]
    assert block.revisions[-1].content == truncated


def test_text_vlm_only_accepts_agreement_or_verified_addition() -> None:
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = SimpleNamespace(vlm_model="test-vlm")
    agreed = _block("doc", 1, "中国人民银行发布报告。", [0, 0, 100, 20])
    pipeline._apply_vlm_result(
        agreed,
        VLMResult(task="ocr_verify", readable=True, text="中国人民银行发布报告。"),
    )
    assert agreed.content == agreed.raw_content
    assert agreed.source == "mineru+qwen_agreed"
    assert agreed.status == ResolutionStatus.ACCEPTED

    addition = _block("doc", 1, "中国人民银行发布报告。", [0, 0, 100, 20])
    addition.metadata["pdf_text_comparison"] = {"reference_excerpt": "标题：中国人民银行发布报告。"}
    pipeline._apply_vlm_result(
        addition,
        VLMResult(task="ocr_verify", readable=True, text="标题：中国人民银行发布报告。"),
    )
    assert addition.content == "标题：中国人民银行发布报告。"
    assert addition.source == "qwen+pdf_text_layer_agreed_addition"
    assert addition.status == ResolutionStatus.REPAIRED
    assert addition.metadata["chunk_eligible"] is True


def test_mineru_run_metadata_round_trip_preserves_real_batch(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    submission = tmp_path / "submission.pdf"
    archive = tmp_path / "mineru" / "mineru-result.zip"
    extracted = tmp_path / "mineru" / "extracted"
    archive.parent.mkdir(parents=True)
    extracted.mkdir(parents=True)
    source.write_bytes(b"source-pdf")
    submission.write_bytes(b"submission-pdf")
    archive.write_bytes(b"zip-result")

    written = _write_mineru_run_metadata(
        tmp_path,
        batch_id="batch-real-123",
        archive_path=archive,
        extracted_dir=extracted,
        source_pdf=source,
        submission_pdf=submission,
        data_id="fresh-nonce",
    )

    assert written["batch_id"] == "batch-real-123"
    assert written["source_sha256"] != written["submission_sha256"]
    assert written["archive_sha256"]
    assert _read_mineru_run_metadata(tmp_path) == written


def test_coloured_table_tiles_snap_to_light_row_rules_and_keep_final_row(tmp_path: Path) -> None:
    source = tmp_path / "coloured-table.png"
    image = Image.new("RGB", (600, 900), "#d9e8f5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 599, 49), fill="#123f73")
    for y in [50, 200, 350, 500, 650, 780]:
        draw.line((0, y, 599, y), fill="white", width=2)
    image.save(source)
    structure = {
        # Deliberately inaccurate VLM regions: the local row-rule detector
        # must correct the header and prevent the last row from being cut.
        "rotation_degrees": 0,
        "header_region": [0, 0, 1, 0.15],
        "data_region": [0, 0.15, 1, 0.87],
    }

    paths = create_table_tiles(
        source,
        tmp_path / "tiles",
        structure,
        max_tile_height=550,
        overlap=100,
    )

    assert len(paths) == 3
    assert [Image.open(path).height for path in paths] == [500, 480, 300]
