from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


_BAD_SECTION_RE = re.compile(r"^(?:单位|币种)[：:]|^(?:续)?(?:附表|表|图)\s*\d+|^续表$")
_FAILURE_PLACEHOLDER_RE = re.compile(
    r"the\s+(?:image|text).*(?:too\s+blurry|cannot|unable).*(?:recognize|read)|"
    r"(?:图片|图像|文字).*(?:太模糊|无法|不能).*(?:识别|读取)",
    re.I | re.S,
)


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def audit_chunk_export(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = list(payload.get("chunks") or [])
    parameters = payload.get("parameters") or {}
    text_target = int(parameters.get("text_size_chars") or 1200)
    hard_failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def fail(code: str, count: int, details: Any = None) -> None:
        if count:
            hard_failures.append({"code": code, "count": count, "details": details})

    def warn(code: str, count: int, details: Any = None) -> None:
        if count:
            warnings.append({"code": code, "count": count, "details": details})

    validation = payload.get("validation") or {}
    fail("schema_not_v3", int(payload.get("schema_version") != "multimodal-chunks-v3"), payload.get("schema_version"))
    fail("export_validation_failed", sum(value is not True for value in validation.values()), validation)
    fail("empty_chunk_set", int(not chunks))

    ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    fail("duplicate_chunk_id", len(ids) - len(set(ids)))
    fail("empty_chunk_text", sum(not str(chunk.get("display_text") or "").strip() for chunk in chunks))
    fail("missing_source_blocks", sum(not (chunk.get("metadata") or {}).get("source_block_ids") for chunk in chunks))

    page_start_regressions = [
        index for index in range(1, len(chunks))
        if int(chunks[index].get("page_start") or 0) < int(chunks[index - 1].get("page_start") or 0)
    ]
    page_end_regressions = [
        index for index in range(1, len(chunks))
        if int(chunks[index].get("page_end") or 0) < int(chunks[index - 1].get("page_end") or 0)
    ]
    fail("page_start_regression", len(page_start_regressions), page_start_regressions[:20])
    fail("page_end_regression", len(page_end_regressions), page_end_regressions[:20])

    normalized_texts: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", str(chunk.get("display_text") or "")).strip()
        normalized_texts[normalized].append(str(chunk.get("chunk_id") or ""))
    exact_duplicate_groups = {
        text: chunk_ids for text, chunk_ids in normalized_texts.items() if text and len(chunk_ids) > 1
    }
    fail("exact_duplicate_text", len(exact_duplicate_groups), list(exact_duplicate_groups.items())[:20])

    bad_sections = []
    provisional_missing_tags = []
    missing_order_keys = []
    invalid_table_headers = []
    failure_placeholders = []
    text_over_target = []
    embedding_lengths: list[int] = []
    role_lengths: dict[str, list[int]] = defaultdict(list)
    tiers: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        chunk_id = str(chunk.get("chunk_id") or "")
        role = str(metadata.get("chunk_role") or "unknown")
        tier = str(metadata.get("retrieval_tier") or "")
        text = str(chunk.get("display_text") or "")
        embedding = str(chunk.get("embedding_text") or "")
        roles[role] += 1
        tiers[tier] += 1
        modalities[str(chunk.get("block_type") or "unknown")] += 1
        embedding_lengths.append(len(embedding))
        role_lengths[role].append(len(text))
        for value in metadata.get("section_path") or []:
            if _BAD_SECTION_RE.search(str(value).strip()):
                bad_sections.append({"chunk_id": chunk_id, "section": value})
        if tier == "provisional" and (
            metadata.get("requires_human_review") is not True
            or "mineru_raw" not in set((metadata.get("source_content_basis") or {}).values())
        ):
            provisional_missing_tags.append(chunk_id)
        if not all(key in metadata for key in ("retrieval_order", "anchor_page", "anchor_reading_order", "article_order_key")):
            missing_order_keys.append(chunk_id)
        headers = list(metadata.get("header_paths") or [])
        cells = metadata.get("cells") or {}
        if headers and len(headers) != len(set(headers)):
            invalid_table_headers.append({"chunk_id": chunk_id, "reason": "duplicate_headers"})
        if role == "table_row" and headers and isinstance(cells, dict) and len(cells) != len(headers):
            invalid_table_headers.append({"chunk_id": chunk_id, "reason": "cell_header_count_mismatch"})
        if len(text) <= 240 and _FAILURE_PLACEHOLDER_RE.search(text):
            failure_placeholders.append(chunk_id)
        if role == "text_segment" and len(text) > text_target:
            text_over_target.append({"chunk_id": chunk_id, "length": len(text)})

    fail("bad_section_path", len(bad_sections), bad_sections[:20])
    fail("provisional_missing_quality_tags", len(provisional_missing_tags), provisional_missing_tags[:20])
    fail("missing_order_metadata", len(missing_order_keys), missing_order_keys[:20])
    fail("invalid_table_header_mapping", len(invalid_table_headers), invalid_table_headers[:20])
    fail("model_failure_placeholder", len(failure_placeholders), failure_placeholders[:20])
    fail("text_chunk_over_target", len(text_over_target), text_over_target[:20])

    short_text = [
        chunk for chunk in chunks
        if (chunk.get("metadata") or {}).get("chunk_role") == "text_segment"
        and len(str(chunk.get("display_text") or "")) < 20
    ]
    warn("short_text_chunk", len(short_text), [
        {"page": chunk.get("page_start"), "text": chunk.get("display_text")}
        for chunk in short_text[:30]
    ])
    long_non_text = [
        {"chunk_id": chunk.get("chunk_id"), "role": (chunk.get("metadata") or {}).get("chunk_role"), "length": len(str(chunk.get("embedding_text") or ""))}
        for chunk in chunks
        if len(str(chunk.get("embedding_text") or "")) > max(1600, text_target * 2)
    ]
    warn("long_non_text_chunk", len(long_non_text), long_non_text[:20])

    known_checks: dict[str, bool] = {}
    document_id = str((payload.get("document") or {}).get("document_id") or "")
    if document_id == "4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393":
        appendix_summaries = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("chunk_role") == "table_summary"
            and 100 <= int(chunk.get("page_start") or 0) <= 107
        ]
        known_checks = {
            "eight_independent_tables_pages_100_107": (
                len(appendix_summaries) == 8
                and all(chunk.get("page_start") == chunk.get("page_end") for chunk in appendix_summaries)
            ),
            "page_56_full_mineru_sentence_represented": any(
                int(chunk.get("page_start") or 0) == 56
                and "质押式回购利率为1.79" in str(chunk.get("display_text") or "")
                for chunk in chunks
            ),
            "page_115_tables_represented": sum(
                int(chunk.get("page_start") or 0) == 115 and chunk.get("block_type") == "table"
                for chunk in chunks
            ) >= 2,
            "page_84_first_scenario_row_represented": any(
                int(chunk.get("page_start") or 0) == 84
                and (chunk.get("metadata") or {}).get("chunk_role") == "table_row"
                and "情景1" in str(chunk.get("display_text") or "")
                for chunk in chunks
            ),
        }
        fail("known_regression_failed", sum(not value for value in known_checks.values()), known_checks)

    metrics = {
        "chunk_count": len(chunks),
        "logical_parent_count": len({
            (chunk.get("metadata") or {}).get("logical_parent_id") for chunk in chunks
        }),
        "role_counts": dict(sorted(roles.items())),
        "modality_counts": dict(sorted(modalities.items())),
        "retrieval_tier_counts": dict(sorted(tiers.items())),
        "embedding_length_chars": {
            "min": min(embedding_lengths, default=0),
            "median": int(median(embedding_lengths)) if embedding_lengths else 0,
            "p95": _percentile(embedding_lengths, 0.95),
            "max": max(embedding_lengths, default=0),
        },
        "role_length_chars": {
            role: {
                "count": len(values),
                "median": int(median(values)),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
            for role, values in sorted(role_lengths.items())
        },
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "page_start_regression_count": len(page_start_regressions),
        "page_end_regression_count": len(page_end_regressions),
        "bad_section_path_count": len(bad_sections),
        "invalid_table_header_mapping_count": len(invalid_table_headers),
        "short_text_chunk_count": len(short_text),
        "known_document_checks": known_checks,
    }
    return {
        "status": "passed" if not hard_failures else "failed",
        "score": max(0, 100 - 20 * len(hard_failures) - min(20, len(warnings))),
        "input": str(path.resolve()),
        "document_id": document_id,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "metrics": metrics,
    }


def write_chunk_quality_report(result: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    lines = [
        "# Chunk 质量门控报告",
        "",
        f"- 状态：`{result['status']}`",
        f"- 质量分：`{result['score']}/100`",
        f"- Chunk 数：`{metrics['chunk_count']}`",
        f"- 精确重复组：`{metrics['exact_duplicate_group_count']}`",
        f"- 页码倒退：start=`{metrics['page_start_regression_count']}`，end=`{metrics['page_end_regression_count']}`",
        f"- 污染章节路径：`{metrics['bad_section_path_count']}`",
        f"- 表头/单元格映射错误：`{metrics['invalid_table_header_mapping_count']}`",
        "",
        "## 长度统计",
        "",
        "```json",
        json.dumps(metrics["embedding_length_chars"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 模态与质量层级",
        "",
        "```json",
        json.dumps({
            "modalities": metrics["modality_counts"],
            "roles": metrics["role_counts"],
            "retrieval_tiers": metrics["retrieval_tier_counts"],
        }, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 已知问题回归检查",
        "",
    ]
    for name, passed in metrics["known_document_checks"].items():
        lines.append(f"- {'通过' if passed else '失败'}：`{name}`")
    lines.extend(["", "## 硬失败", ""])
    lines.append("无。" if not result["hard_failures"] else "```json\n" + json.dumps(result["hard_failures"], ensure_ascii=False, indent=2) + "\n```")
    lines.extend(["", "## 警告", ""])
    lines.append("无。" if not result["warnings"] else "```json\n" + json.dumps(result["warnings"], ensure_ascii=False, indent=2) + "\n```")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
