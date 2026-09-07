from __future__ import annotations

from pathlib import Path

from mineru_vlm_rag.chunking import build_chunks, write_chunk_export
from mineru_vlm_rag.domain.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
    ResolutionStatus,
)
from mineru_vlm_rag.persistence import MySQLRepository


def _block(
    document_id: str,
    page_no: int,
    content: str,
    block_type: BlockType = BlockType.TEXT,
    *,
    column: int | None = None,
) -> Block:
    metadata = {"column_index": column} if column is not None else {}
    return Block(
        document_id=document_id,
        page_no=page_no,
        block_type=block_type,
        bbox=BoundingBox(x0=10, y0=10, x1=900, y1=100),
        raw_content=content,
        resolved_content=content,
        section_path=["第一章", "第一节"],
        metadata=metadata,
    )


def _document(blocks: list[Block], page_count: int = 1) -> Document:
    pages = [
        Page(
            document_id="doc",
            page_no=page_no,
            width=1000,
            height=1400,
            blocks=[block for block in blocks if block.page_no == page_no],
        )
        for page_no in range(1, page_count + 1)
    ]
    for page in pages:
        for order, block in enumerate(page.blocks):
            block.reading_order = order
    return Document(
        document_id="doc",
        file_name="complex.pdf",
        source_path="/tmp/complex.pdf",
        sha256="a" * 64,
        page_count=page_count,
        pages=pages,
    )


def test_text_chunks_merge_adjacent_blocks_but_not_columns_and_tier_review() -> None:
    left_a = _block("doc", 1, "第一段内容完整。", column=0)
    left_b = _block("doc", 1, "第二段与第一段属于同一语义区域。", column=0)
    right = _block("doc", 1, "右栏内容不得与左栏合并。", column=1)
    rejected = _block("doc", 1, "待复核内容不得进入Chunk。", column=1)
    rejected.status = ResolutionStatus.REVIEW

    chunks = build_chunks(_document([left_a, left_b, right, rejected]), text_size=500, overlap=50)

    assert [chunk.metadata["chunk_role"] for chunk in chunks] == [
        "text_segment", "text_segment", "text_segment",
    ]
    assert chunks[0].metadata["source_block_ids"] == [left_a.block_id, left_b.block_id]
    assert chunks[1].metadata["source_block_ids"] == [right.block_id]
    review_chunk = chunks[2]
    assert review_chunk.metadata["source_block_ids"] == [rejected.block_id]
    assert review_chunk.metadata["retrieval_tier"] == "provisional"
    assert review_chunk.metadata["source_content_basis"][rejected.block_id] == "mineru_raw"
    assert "待复核" in review_chunk.embedding_text
    assert chunks[0].metadata["next_chunk_id"] == chunks[1].chunk_id
    assert chunks[1].metadata["previous_chunk_id"] == chunks[0].chunk_id


def test_table_chunks_bind_context_and_emit_summary_groups_and_exact_rows() -> None:
    rows = "".join(
        f"<tr><td>项目{index}</td><td>{index}.25</td><td>{index}%</td></tr>"
        for index in range(1, 8)
    )
    table = _block(
        "doc",
        1,
        "<table><thead>"
        "<tr><th rowspan='2'>项目</th><th colspan='2'>2025年</th></tr>"
        "<tr><th>金额</th><th>同比</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>",
        BlockType.TABLE,
    )
    title = _block("doc", 1, "金融业资产简表", BlockType.TITLE)
    title.metadata["associated_table_block_id"] = table.block_id
    unit = _block("doc", 1, "单位：万亿元")
    unit.metadata["associated_table_block_id"] = table.block_id

    chunks = build_chunks(
        _document([title, unit, table]),
        text_size=1200,
        overlap=120,
        table_row_group_size=3,
    )
    roles = [chunk.metadata["chunk_role"] for chunk in chunks]

    assert roles.count("table_summary") == 1
    assert roles.count("table_row_group") == 2
    assert roles.count("table_row") == 7
    assert len({chunk.display_text for chunk in chunks}) == len(chunks)
    assert "表题：金融业资产简表" in chunks[0].display_text
    assert "单位：万亿元" in chunks[0].display_text
    exact = next(chunk for chunk in chunks if chunk.metadata["chunk_role"] == "table_row")
    assert exact.metadata["header_paths"] == ["项目", "2025年 / 金额", "2025年 / 同比"]
    assert exact.parent_id == chunks[0].chunk_id
    assert set(exact.metadata["source_block_ids"]) == {table.block_id, title.block_id, unit.block_id}
    assert all(chunk.metadata["chunk_role"] != "text_segment" for chunk in chunks)


def test_chart_facts_formula_context_and_json_export(tmp_path: Path) -> None:
    before = _block("doc", 1, "其中资本充足率计算如下。")
    formula = _block("doc", 1, "CAR=资本净额/风险加权资产", BlockType.FORMULA)
    after = _block("doc", 1, "该指标用于衡量银行风险抵补能力。")
    chart = _block(
        "doc",
        1,
        "2023年指标为10.2%。2024年上升至11.5%。总体趋势保持增长。",
        BlockType.CHART,
    )
    chart.metadata["caption"] = "资本充足率变化"
    document = _document([before, formula, after, chart])

    chunks = build_chunks(document, text_size=500, overlap=50)
    roles = [chunk.metadata["chunk_role"] for chunk in chunks]

    assert "formula_context" in roles
    formula_chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_role"] == "formula_context")
    assert "前文：其中资本充足率计算如下。" in formula_chunk.display_text
    assert "后文：该指标用于衡量银行风险抵补能力。" in formula_chunk.display_text
    assert roles.count("chart_summary") == 1
    assert roles.count("chart_fact") == 3

    output = tmp_path / "all_chunks.json"
    payload = write_chunk_export(
        document,
        chunks,
        output,
        parameters={"text_size_chars": 500, "text_overlap_chars": 50, "table_row_group_size": 6},
    )
    assert output.exists()
    assert payload["embedding_executed"] is False
    assert payload["database_chunks_replaced"] is False
    assert payload["summary"]["chunk_count"] == len(chunks)
    assert len(payload["chunks"]) == len(chunks)
    assert all(payload["validation"].values())
    assert payload["summary"]["source_block_count"] == 4


def test_mysql_document_graph_can_be_loaded_for_offline_rechunk(tmp_path: Path) -> None:
    source = _block("doc", 1, "只从数据库恢复并切块。")
    document = _document([source])
    initial_chunks = build_chunks(document)
    repository = MySQLRepository(f"sqlite+pysqlite:///{tmp_path / 'graph.db'}")
    try:
        repository.initialize()
        repository.save_document_graph(document, initial_chunks)
        loaded = repository.load_document_graph("doc")
    finally:
        repository.engine.dispose()

    assert loaded.document_id == document.document_id
    assert loaded.pages[0].blocks[0].block_id == source.block_id
    assert loaded.pages[0].blocks[0].content == source.content
    assert build_chunks(loaded)[0].metadata["source_block_ids"] == [source.block_id]


def test_mysql_can_replace_only_chunks_after_offline_index(tmp_path: Path) -> None:
    source = _block("doc", 1, "原始证据块不会被离线索引删除。")
    document = _document([source])
    original = build_chunks(document)
    repository = MySQLRepository(f"sqlite+pysqlite:///{tmp_path / 'replace.db'}")
    try:
        repository.initialize()
        repository.save_document_graph(document, original)
        replacement = build_chunks(document, text_size=30, overlap=3)
        count = repository.replace_document_chunks("doc", replacement, embedding_state="embedded")
        loaded = repository.load_document_graph("doc")
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from mineru_vlm_rag.persistence.mysql_repository import ChunkRow
        with Session(repository.engine) as session:
            rows = list(session.scalars(select(ChunkRow).where(ChunkRow.document_id == "doc")))
    finally:
        repository.engine.dispose()

    assert count == len(replacement)
    assert loaded.pages[0].blocks[0].content == source.content
    assert {row.chunk_id for row in rows} == {chunk.chunk_id for chunk in replacement}
    assert all(row.embedding_state == "embedded" for row in rows)
