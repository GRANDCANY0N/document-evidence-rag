from __future__ import annotations

from mineru_vlm_rag.evaluation.retrieval_quality import _keyword_hit, _prefix_metrics
from mineru_vlm_rag.pipeline.query import QueryService


def test_table_level_unit_can_validate_a_numeric_cell() -> None:
    text = "单位：亿元，%\n数据行：2024=56963.10；同比增长=5.68"

    assert _keyword_hit("56963.10亿元", text)
    assert _keyword_hit("5.68%", text)
    assert not _keyword_hit("56963.11亿元", text)
    assert not _keyword_hit("134.9万亿元", "单位：万亿元；国内生产总值=134.91")


def test_prefix_metrics_union_multiple_rows_on_the_same_page() -> None:
    question = {
        "page_no": 101,
        "page_range": None,
        "must_retrieve_keywords": ["总资产", "总负债", "440513.31亿元"],
    }
    results = [
        {"page_start": 101, "page_end": 101, "text": "单位：亿元；总资产=440513.31"},
        {"page_start": 101, "page_end": 101, "text": "单位：亿元；总负债=440513.31"},
    ]

    metric = _prefix_metrics(question, results, 2)

    assert metric["page_hit"] is True
    assert metric["strict_keyword_success"] is True
    assert metric["best_single_chunk_keyword_coverage"] < metric["keyword_coverage"]


def test_character_terms_cover_chinese_phrase_and_exact_number() -> None:
    terms = QueryService._lexical_terms("超过1900家专精特新企业已在A股上市")

    assert "1900" in terms
    assert "专精" in terms
    assert "上市" in terms


def test_final_rank_stabilizes_near_tied_table_rows() -> None:
    # A row ranked third by the API but first by both retrieval channels
    # should beat an almost-tied row that all retrieval channels rank lower.
    requested_row = QueryService._final_rank_score(
        relevance_score=0.973,
        dense_rank=1,
        lexical_rank=1,
    )
    neighbouring_row = QueryService._final_rank_score(
        relevance_score=0.980,
        dense_rank=6,
        lexical_rank=8,
    )

    assert requested_row > neighbouring_row
