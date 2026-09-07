from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = value.replace("\\%", "%").replace("−", "-").replace("—", "-")
    return re.sub(r"[\s`*_{}$^，。；：、（）()\[\]《》<>\"'“”‘’]+", "", value)


def _keyword_hit(keyword: str, text: str) -> bool:
    needle = _normalize(keyword)
    haystack = _normalize(text)
    if not needle:
        return False
    if needle in haystack:
        return True
    # Table cells store raw numbers while the unit is carried once in the
    # table context.  Treat ``56963.10`` + ``单位：亿元`` as evidence for
    # keyword ``56963.10亿元`` without rewriting the canonical table row.
    numeric_unit = re.search(
        r"(-?\d+(?:\.\d+)?)(万亿元|亿元|亿笔|万笔|亿美元|万亿美元|个百分点|%|家|只)$",
        needle,
    )
    if numeric_unit and numeric_unit.start() == 0:
        number, unit = numeric_unit.groups()
        number_pattern = re.compile(rf"(?<![\d.]){re.escape(number)}(?![\d.])")
        if number_pattern.search(haystack) and unit in haystack:
            return True
    return False


def _target_pages(question: dict[str, Any]) -> set[int]:
    if question.get("page_no") is not None:
        return {int(question["page_no"])}
    page_range = question.get("page_range") or []
    if len(page_range) == 2:
        return set(range(int(page_range[0]), int(page_range[1]) + 1))
    return set()


def _result_pages(result: dict[str, Any]) -> set[int]:
    start = int(result.get("page_start") or 0)
    end = int(result.get("page_end") or start)
    return set(range(start, end + 1)) if start else set()


def _prefix_metrics(question: dict[str, Any], results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    prefix = results[:k]
    keywords = [str(value) for value in (question.get("must_retrieve_keywords") or [])]
    target_pages = _target_pages(question)
    union_text = "\n".join(str(result.get("text") or "") for result in prefix)
    hits = [keyword for keyword in keywords if _keyword_hit(keyword, union_text)]
    best_single = max(
        (
            sum(_keyword_hit(keyword, str(result.get("text") or "")) for keyword in keywords)
            for result in prefix
        ),
        default=0,
    )
    page_hit = any(_result_pages(result) & target_pages for result in prefix) if target_pages else None
    coverage = len(hits) / len(keywords) if keywords else 1.0
    return {
        "k": k,
        "page_hit": page_hit,
        "keyword_hits": hits,
        "keyword_hit_count": len(hits),
        "keyword_count": len(keywords),
        "keyword_coverage": round(coverage, 4),
        "best_single_chunk_keyword_coverage": round(best_single / len(keywords), 4) if keywords else 1.0,
        "content_success": coverage >= 0.60,
        "strict_keyword_success": len(hits) == len(keywords),
        "grounded_success": coverage >= 0.60 and page_hit is not False,
    }


def _aggregate(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    def group_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"question_count": len(values)}
        for k in ks:
            key = f"at_{k}"
            metrics = [value["metrics"][key] for value in values]
            output[key] = {
                "page_hit_rate": round(
                    sum(metric["page_hit"] is True for metric in metrics)
                    / max(1, sum(metric["page_hit"] is not None for metric in metrics)),
                    4,
                ),
                "content_success_rate": round(
                    sum(metric["content_success"] for metric in metrics) / max(1, len(metrics)), 4
                ),
                "grounded_success_rate": round(
                    sum(metric["grounded_success"] for metric in metrics) / max(1, len(metrics)), 4
                ),
                "strict_keyword_success_rate": round(
                    sum(metric["strict_keyword_success"] for metric in metrics) / max(1, len(metrics)), 4
                ),
                "mean_keyword_coverage": round(
                    sum(metric["keyword_coverage"] for metric in metrics) / max(1, len(metrics)), 4
                ),
            }
        return output

    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_modality[str(row["modality"])].append(row)
        by_difficulty[str(row["difficulty"])].append(row)
    return {
        "overall": group_metrics(rows),
        "by_modality": {key: group_metrics(values) for key, values in sorted(by_modality.items())},
        "by_difficulty": {key: group_metrics(values) for key, values in sorted(by_difficulty.items())},
    }


def evaluate_retrieval(
    questions_path: Path,
    search: Callable[..., list[dict[str, Any]]],
    *,
    document_id: str,
    output_dir: Path,
    limit: int = 10,
) -> dict[str, Any]:
    """Evaluate actual dense+rerank retrieval against PDF-only questions."""
    dataset = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = list(dataset.get("questions") or [])
    if not questions:
        raise ValueError("Question set is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = tuple(value for value in (1, 3, 5, 10) if value <= limit)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, question in enumerate(questions, start=1):
        LOGGER.info("[召回评测] start question=%s/%s id=%s", index, len(questions), question.get("id"))
        last_error: Exception | None = None
        results: list[dict[str, Any]] = []
        for attempt in range(2):
            try:
                results = search(
                    str(question["question"]),
                    document_id=document_id,
                    limit=limit,
                )
                last_error = None
                break
            except Exception as exc:  # one bounded retry for transient provider errors
                last_error = exc
                if attempt == 0:
                    LOGGER.warning(
                        "[召回评测] retry id=%s error=%s",
                        question.get("id"), type(exc).__name__,
                    )
                    time.sleep(1.0)
        if last_error is not None:
            raise RuntimeError(f"Retrieval failed for {question.get('id')}: {last_error}") from last_error
        metrics = {f"at_{k}": _prefix_metrics(question, results, k) for k in ks}
        row = {
            "id": question.get("id"),
            "question": question.get("question"),
            "expected_answer": question.get("expected_answer"),
            "page_no": question.get("page_no"),
            "page_range": question.get("page_range"),
            "modality": question.get("modality"),
            "difficulty": question.get("difficulty"),
            "must_retrieve_keywords": question.get("must_retrieve_keywords") or [],
            "metrics": metrics,
            "results": results,
        }
        rows.append(row)
        checkpoint = {
            "status": "running",
            "document_id": document_id,
            "completed": len(rows),
            "total": len(questions),
            "rows": rows,
        }
        (output_dir / "retrieval_eval_checkpoint.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        top = metrics[f"at_{max(ks)}"]
        LOGGER.info(
            "[召回评测] done id=%s page_hit=%s keyword_coverage=%.2f",
            question.get("id"), top["page_hit"], top["keyword_coverage"],
        )

    tier_counts = Counter(
        str(result.get("retrieval_tier") or "unknown")
        for row in rows
        for result in row["results"]
    )
    payload = {
        "status": "completed",
        "document_id": document_id,
        "questions_path": str(questions_path.resolve()),
        "question_count": len(rows),
        "limit": limit,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "metric_definition": {
            "page_hit": "top-k任一chunk页码与PDF标注页/页范围相交",
            "keyword_coverage": "top-k所有文本的must_retrieve_keywords精确归一化命中比例",
            "content_success": "keyword_coverage >= 0.60",
            "grounded_success": "content_success且page_hit不为false",
            "strict_keyword_success": "全部must_retrieve_keywords均命中",
        },
        "retrieved_tier_counts": dict(sorted(tier_counts.items())),
        "summary": _aggregate(rows, ks),
        "rows": rows,
    }
    json_path = output_dir / "retrieval_quality_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, output_dir / "retrieval_quality_report.md")
    return payload


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    overall = payload["summary"]["overall"]
    lines = [
        "# 原 PDF 独立问题召回质量报告",
        "",
        f"- 文档：`{payload['document_id']}`",
        f"- 问题数：{payload['question_count']}",
        f"- 评测耗时：{payload['elapsed_seconds']} 秒",
        "- 说明：问题由子代理只看原 PDF 构造；评测时才查询 chunk 索引。",
        "",
        "## 总体结果",
        "",
        "| k | 页命中率 | 内容成功率 | 页+内容成功率 | 全关键词命中率 | 平均关键词覆盖率 |",
        "| -: | -: | -: | -: | -: | -: |",
    ]
    for key, values in overall.items():
        if not key.startswith("at_"):
            continue
        lines.append(
            f"| {key[3:]} | {values['page_hit_rate']:.1%} | {values['content_success_rate']:.1%} | "
            f"{values['grounded_success_rate']:.1%} | {values['strict_keyword_success_rate']:.1%} | "
            f"{values['mean_keyword_coverage']:.1%} |"
        )
    lines.extend(["", "## 按模态（Top-10）", "", "| 模态 | 数量 | 页命中率 | 内容成功率 | 页+内容成功率 | 平均关键词覆盖率 |", "| --- | -: | -: | -: | -: | -: |"]) 
    for modality, group in payload["summary"]["by_modality"].items():
        values = group.get("at_10") or group[next(key for key in group if key.startswith("at_"))]
        lines.append(
            f"| {modality} | {group['question_count']} | {values['page_hit_rate']:.1%} | "
            f"{values['content_success_rate']:.1%} | {values['grounded_success_rate']:.1%} | "
            f"{values['mean_keyword_coverage']:.1%} |"
        )
    lines.extend(["", "## 逐题结果（Top-10）", "", "| ID | 模态 | 难度 | 页命中 | 关键词覆盖 | 结论 | Top-1页/角色/质量层 |", "| --- | --- | --- | --- | -: | --- | --- |"]) 
    for row in payload["rows"]:
        metric = row["metrics"].get("at_10") or list(row["metrics"].values())[-1]
        top = row["results"][0] if row["results"] else {}
        top_value = f"p{top.get('page_start', '-')}/{top.get('chunk_role', '-')}/{top.get('retrieval_tier', '-')}"
        conclusion = "通过" if metric["grounded_success"] else "需分析"
        lines.append(
            f"| {row['id']} | {row['modality']} | {row['difficulty']} | {metric['page_hit']} | "
            f"{metric['keyword_coverage']:.1%} | {conclusion} | {top_value} |"
        )
    lines.extend(["", "## 未通过题目明细", ""])
    failures = [
        row for row in payload["rows"]
        if not (row["metrics"].get("at_10") or list(row["metrics"].values())[-1])["grounded_success"]
    ]
    if not failures:
        lines.append("无。")
    for row in failures:
        metric = row["metrics"].get("at_10") or list(row["metrics"].values())[-1]
        missed = [
            value for value in row["must_retrieve_keywords"]
            if value not in metric["keyword_hits"]
        ]
        lines.extend([
            f"### {row['id']} {row['question']}",
            "",
            f"- 标准答案：{row['expected_answer']}",
            f"- 页命中：{metric['page_hit']}；关键词覆盖：{metric['keyword_coverage']:.1%}",
            f"- 未命中关键词：{', '.join(missed) or '无'}",
            "- Top-3：" + "；".join(
                f"p{result.get('page_start')}/{result.get('chunk_role')}: "
                f"{str(result.get('text') or '').replace(chr(10), ' ')[:120]}"
                for result in row["results"][:3]
            ),
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
