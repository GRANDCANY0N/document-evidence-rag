from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mineru_vlm_rag.domain.models import BoundingBox, Document
from mineru_vlm_rag.normalization.mineru_parser import parse_mineru_output
from mineru_vlm_rag.persistence.mysql_repository import (
    AssetRow,
    BlockRow,
    ChunkRow,
    DocumentRow,
    MySQLRepository,
    RevisionRow,
)
from mineru_vlm_rag.quality.completeness import (
    _char_multiset_f1,
    _block_reference,
    extract_pdf_text_layer,
    flatten_content,
    normalize_text,
    number_tokens,
)
from mineru_vlm_rag.settings import load_settings


SEMANTIC_TYPES = {"text", "title", "footnote", "table"}
TEMPLATE_TYPES = {"header", "footer", "page_number"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a complete post-processing result report")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _md(value: Any, *, empty: str = "（空）") -> str:
    if value is None or value == "":
        return empty
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = html.escape(str(value), quote=False).replace("|", "&#124;")
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def _link(path: str | Path, label: str = "证据") -> str:
    path = str(path)
    return f"[{label}](<{path}>)" if path else "（无文件证据）"


def _bbox(values: list[float] | None) -> BoundingBox | None:
    if not values or len(values) != 4:
        return None
    return BoundingBox(x0=float(values[0]), y0=float(values[1]), x1=float(values[2]), y1=float(values[3]))


def _covers(box: BoundingBox | None, line: Any) -> bool:
    if box is None:
        return False
    cx = (line.bbox.x0 + line.bbox.x1) / 2.0
    cy = (line.bbox.y0 + line.bbox.y1) / 2.0
    inside = box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1
    x = max(0.0, min(box.x1, line.bbox.x1) - max(box.x0, line.bbox.x0))
    y = max(0.0, min(box.y1, line.bbox.y1) - max(box.y0, line.bbox.y0))
    ratio = x * y / max(1e-9, line.bbox.width * line.bbox.height)
    return inside or ratio >= 0.35


def _source_text(page_no: int, bbox: list[float] | None, text_layer: Any) -> str:
    box = _bbox(bbox)
    lines = [line for line in text_layer.pages.get(page_no, []) if _covers(box, line)]
    lines.sort(key=lambda line: (line.line_no, line.bbox.x0))
    return "\n".join(line.text for line in lines)


def _evidence(events: list[dict[str, Any]], block_id: str) -> str:
    paths: list[str] = []
    for event in events:
        if event.get("block_id") != block_id:
            continue
        for path in event.get("input_refs") or []:
            if isinstance(path, str) and path.startswith("/") and path not in paths:
                paths.append(path)
    return "<br>".join(_link(path, f"证据{index}") for index, path in enumerate(paths, start=1)) or "（无独立图片）"


def _page_evidence(work_dir: Path, page_no: int) -> str:
    path = work_dir / "pages" / f"page-{page_no:04d}.png"
    return _link(path, "原页图") if path.exists() else "（无原页图）"


def _has_duplicate_insertion(raw_content: str | None, resolved_content: str | None) -> bool:
    raw = normalize_text(flatten_content(raw_content or ""))
    resolved = normalize_text(flatten_content(resolved_content or ""))
    return any(
        tag == "insert"
        and j2 - j1 >= 2
        and resolved.count(resolved[j1:j2]) > raw.count(resolved[j1:j2])
        for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, raw, resolved, autojunk=False).get_opcodes()
    )


def _strong_truncation(raw_content: str | None, resolved_content: str | None) -> bool:
    raw_text = (raw_content or "").strip()
    resolved_text = (resolved_content or "").strip()
    raw = re.sub(r"[$\\{}^]", "", unicodedata.normalize("NFKC", normalize_text(raw_text)))
    resolved = re.sub(r"[$\\{}^]", "", unicodedata.normalize("NFKC", normalize_text(resolved_text)))
    return (
        raw_text.endswith(("。", "！", "？", "；"))
        and not resolved_text.endswith(("。", "！", "？", "；"))
        and len(raw) > len(resolved) + 5
    )


def _line_match(line: Any, blocks: list[Any], *, final: bool) -> bool:
    target = line.normalized
    values: list[str] = []
    for block in blocks:
        content = block.resolved_content if final else block.content
        if content and content.strip():
            values.append(normalize_text(flatten_content(content)))
    if any(target in value or _char_multiset_f1(target, value) >= 0.92 for value in values):
        return True
    return bool(values) and target in "".join(values)


def _raw_type(block: Any) -> str:
    value = block.block_type
    return value.value if hasattr(value, "value") else str(value)


def _strict_audit(raw_document: Document, final_blocks: dict[int, list[BlockRow]], text_layer: Any) -> dict[str, int]:
    raw_pages = {page.page_no: page.blocks for page in raw_document.pages}
    counts: Counter[str] = Counter()
    for page_no, lines in text_layer.pages.items():
        raw_blocks = raw_pages.get(page_no, [])
        stored_blocks = final_blocks.get(page_no, [])
        for line in lines:
            if len(line.normalized) < 4:
                continue
            counts["all_lines"] += 1
            if any(_raw_type(block) in TEMPLATE_TYPES and _covers(block.bbox, line) for block in raw_blocks):
                counts["template_lines"] += 1
                continue
            counts["content_lines"] += 1
            raw_candidates = [
                block for block in raw_blocks
                if _raw_type(block) in SEMANTIC_TYPES and _covers(block.bbox, line)
            ]
            final_candidates = [
                block for block in stored_blocks
                if block.block_type in SEMANTIC_TYPES and _covers(_bbox(block.bbox_json), line)
            ]
            if raw_candidates:
                counts["raw_geometric"] += 1
            if final_candidates:
                counts["final_geometric"] += 1
            if _line_match(line, raw_candidates, final=False):
                counts["raw_matched"] += 1
            if _line_match(line, final_candidates, final=True):
                counts["final_matched"] += 1
    return dict(counts)


def _pct(value: int, total: int) -> str:
    return f"{value / total * 100:.3f}%" if total else "0.000%"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _same_raw_block(raw_document: Document, stored: BlockRow) -> Any | None:
    candidates = [
        block for page in raw_document.pages if page.page_no == stored.page_no
        for block in page.blocks
        if _raw_type(block) == stored.block_type and block.bbox
    ]
    if not stored.bbox_json:
        return None
    return next(
        (
            block for block in candidates
            if all(abs(left - right) <= 0.01 for left, right in zip(block.bbox.as_list(), stored.bbox_json))
        ),
        None,
    )


def main() -> None:
    args = _args()
    settings = load_settings()
    project_root = settings.project_root
    work_dir = project_root / "outputs" / args.document_id
    audit_path = work_dir / "postprocess" / "page_audit.jsonl"
    audit_summary_path = work_dir / "postprocess" / "page_audit_summary.json"
    workflow_path = project_root / "outputs" / "workflow_logs" / f"{args.document_id}.workflow.jsonl"
    output = args.output or work_dir / "POSTPROCESS_RESULT_REPORT.md"

    findings = _load_jsonl(audit_path)
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    all_events = _load_jsonl(workflow_path)
    events = [event for event in all_events if (event.get("details") or {}).get("run_id") == args.run_id]

    repository = MySQLRepository(settings.mysql_dsn)
    with Session(repository.engine) as session:
        document = session.get(DocumentRow, args.document_id)
        if document is None:
            raise SystemExit(f"Document not found in MySQL: {args.document_id}")
        blocks = list(session.scalars(
            select(BlockRow)
            .where(BlockRow.document_id == args.document_id)
            .order_by(BlockRow.page_no, BlockRow.reading_order)
        ))
        revisions = list(session.scalars(
            select(RevisionRow)
            .where(RevisionRow.block_id.in_([block.block_id for block in blocks]))
            .order_by(RevisionRow.created_at)
        ))
        chunk_count = int(session.scalar(
            select(func.count()).select_from(ChunkRow).where(ChunkRow.document_id == args.document_id)
        ) or 0)
        asset_count = int(session.scalar(
            select(func.count()).select_from(AssetRow).where(AssetRow.document_id == args.document_id)
        ) or 0)
    repository.engine.dispose()

    blocks_by_id = {block.block_id: block for block in blocks}
    blocks_by_page: dict[int, list[BlockRow]] = defaultdict(list)
    for block in blocks:
        blocks_by_page[block.page_no].append(block)
    revisions_by_block: dict[str, list[RevisionRow]] = defaultdict(list)
    for revision in revisions:
        revisions_by_block[revision.block_id].append(revision)

    source_pdf = Path(document.source_path)
    text_layer = extract_pdf_text_layer(source_pdf)
    raw_document = Document(
        document_id=args.document_id,
        file_name=document.file_name,
        source_path=document.source_path,
        sha256=document.sha256,
        page_count=document.page_count,
    )
    parse_mineru_output(raw_document, work_dir / "mineru" / "extracted")
    strict = _strict_audit(raw_document, blocks_by_page, text_layer)
    content_lines = strict["content_lines"]

    finding_counts = Counter(item["kind"] for item in findings)
    status_counts = Counter(block.status for block in blocks)
    source_counts = Counter(block.source for block in blocks)
    vlm_events = [event for event in events if event.get("stage") == "vlm"]
    vlm_counts = Counter(event["decision"] for event in vlm_events)
    unresolved = [block for block in blocks if block.status in {"review", "unreadable", "failed"}]
    table_events = [event for event in events if event.get("stage") == "table_recovery"]
    cross_page_events = [event for event in events if event.get("stage") == "cross_page_table"]
    risky_text_findings: list[tuple[dict[str, Any], BlockRow, list[str]]] = []
    for item in findings:
        if item.get("kind") not in {"block_text_repaired", "block_text_difference_observed"}:
            continue
        block = blocks_by_id.get(item.get("block_id") or "")
        if block is None:
            continue
        compared_content = (
            item.get("reference_excerpt") or ""
            if item.get("kind") == "block_text_difference_observed"
            else block.resolved_content
        )
        reasons: list[str] = []
        if item.get("numeric_match") is False:
            reasons.append("数字/脚注标记不一致")
        if item.get("character_f1") == 1.0 and float(item.get("sequence_similarity") or 1.0) < 0.98:
            reasons.append("字符集合一致但阅读顺序改变")
        if item.get("character_f1") != 1.0 and _has_duplicate_insertion(block.raw_content, compared_content):
            reasons.append("参照候选包含重复文字")
        if _strong_truncation(block.raw_content, compared_content):
            reasons.append("参照候选在半句中截断")
        if reasons:
            risky_text_findings.append((item, block, reasons))

    lines: list[str] = [
        "# 《2025金融稳定报告》完整后处理结果报告",
        "",
        f"生成时间：2026-09-05  ",
        f"文档 ID：`{args.document_id}`  ",
        f"运行 ID：`{args.run_id}`  ",
        f"原 PDF：`{source_pdf}`",
        "",
        "## 1. 报告范围和口径",
        "",
        "本报告只整理后处理阶段发现或改变的内容：页级遗漏、空 block、已有文字修正、视觉区域文字、全部 VLM 调用、复杂表格复核、跨页表判断和全部未解决项。未触发任何问题且原样接受的普通 block 不逐条重复。",
        "",
        "表格中的“原文”由原页图链接和 PDF 文本层参照共同表示。PDF 文本层不是人工视觉真值：字体编码可能把①/②映射成 a/b，双栏、加粗文本也可能重复或错序。因此报告忠实列出程序当时使用的参照，并把可疑的自动修订另行标记。`accepted/repaired` 是流程状态，不等于人工真值准确。",
        "",
        "## 2. 总体结果",
        "",
        "| 项目 | 数量/结果 |",
        "|---|---:|",
        f"| PDF 页数 | {document.page_count} |",
        f"| 原始 MinerU blocks | {sum(len(page.blocks) for page in raw_document.pages)} |",
        f"| 后处理后 blocks | {len(blocks)} |",
        f"| Assets | {asset_count} |",
        f"| Chunks | {chunk_count} |",
        f"| 页级审计发现 | {len(findings)} |",
        f"| 已有 block 文字修正 | {finding_counts['block_text_repaired']} |",
        f"| 已有 block 差异观察（保留 MinerU） | {finding_counts['block_text_difference_observed']} |",
        f"| 高风险文字差异候选 | {len(risky_text_findings)} |",
        f"| 空 block 恢复 | {finding_counts['empty_block_recovered']} |",
        f"| 完全遗漏 block 补回 | {finding_counts['missing_block']} |",
        f"| 视觉区域文字记录 | {finding_counts['visual_region_text']} |",
        f"| VLM block 调用 | {len(vlm_events)} |",
        f"| VLM repaired | {vlm_counts['repaired']} |",
        f"| VLM review | {vlm_counts['review']} |",
        f"| VLM unreadable | {vlm_counts['unreadable']} |",
        f"| 最终未解决 blocks | {len(unresolved)} |",
        "",
        f"最终状态：`{dict(sorted(status_counts.items()))}`。来源：`{dict(sorted(source_counts.items()))}`。",
        "",
        "### 2.1 文本层比对",
        "",
        f"内置条件一致率为 `{audit_summary['text_layer']['matched_line_count']} / {audit_summary['text_layer']['comparable_line_count']} = {audit_summary['text_layer']['agreement_ratio'] * 100:.4f}%`。这个分母只包含已经有可比 MinerU 语义 block 的行，不能代表全文完整率。",
        "",
        f"本报告另以 {content_lines} 条有效文本层内容行为统一分母，排除了 {strict['template_lines']} 条模板行：",
        "",
        "| 统一口径 | 原始 MinerU | 后处理后 |",
        "|---|---:|---:|",
        f"| 同 bbox 有语义 block 覆盖 | {strict['raw_geometric']} / {content_lines} = {_pct(strict['raw_geometric'], content_lines)} | {strict['final_geometric']} / {content_lines} = {_pct(strict['final_geometric'], content_lines)} |",
        f"| 原文严格包含或字符 F1≥0.92 | {strict['raw_matched']} / {content_lines} = {_pct(strict['raw_matched'], content_lines)} | {strict['final_matched']} / {content_lines} = {_pct(strict['final_matched'], content_lines)} |",
        "",
        "严格匹配仍会受 PDF 换行、跨行合并和表格阅读顺序影响，只用于比较处理前后，不能当人工标注准确率。",
        "",
        "## 3. 关键结论",
        "",
        f"- 页级反向审计共处理 {finding_counts['block_text_repaired'] + finding_counts['block_text_difference_observed'] + finding_counts['empty_block_recovered'] + finding_counts['missing_block']} 项程序判定的文字差异或遗漏；另记录 {finding_counts['visual_region_text']} 项位于视觉区域内的文本。",
        f"- 当前有 {finding_counts['block_text_difference_observed']} 项已有 block 差异仅作观察，MinerU 原文未被覆盖；历史自动修正为 {finding_counts['block_text_repaired']} 项。",
        f"- VLM 共处理 {len(vlm_events)} 个 block，其中 {vlm_counts['repaired']} 个完成修复，{vlm_counts['review'] + vlm_counts['unreadable']} 个没有通过门控。",
        "- 第 87 页条形图被错误进入表格复核：第一次 Qwen 返回 chart，但同时返回 HTML table，程序据 `<table>` 误触发二次 MinerU；二次 MinerU仍返回空 content 的 chart。",
        "- 第 115 页两张横向排版表格的 MinerU 原始表格与 PDF 文本层数字完全一致，但 VLM 方向判断为 0°，切片沿错误方向进行，因此后处理结果不可用并被阻断。",
        "",
        "### 3.1 后处理自身的高风险修订",
        "",
        "下列项目表示 MinerU 与参照源存在高风险差异。非破坏版本只记录候选并保留 MinerU；若读取的是旧运行，才可能看到 `block_text_repaired` 和被覆盖的历史结果。",
        "",
        "| 页 | Block | 原页证据 | MinerU 原结果 | 当前最终结果 | 高风险原因 |",
        "|---:|---|---|---|---|---|",
    ]
    for item, block, reasons in risky_text_findings:
        metrics = (
            f"sequence={item.get('sequence_similarity')}; "
            f"char_f1={item.get('character_f1')}; numeric_match={item.get('numeric_match')}"
        )
        lines.append(
            f"| {item['page_no']} | `{block.block_id}` | {_page_evidence(work_dir, item['page_no'])} | "
            f"{_md(block.raw_content)} | {_md(block.resolved_content)} | {_md('；'.join(reasons))}<br>`{metrics}` |"
        )
    lines.extend([
        "",
        "## 4. 全部页级遗漏和文字修正",
        "",
    ])

    labels = {
        "block_text_repaired": "已有 block 内容错误/排版错序，使用 PDF 文本层修正",
        "block_text_difference_observed": "已有 block 与 PDF 文本层不同，仅观察并保留 MinerU",
        "empty_block_recovered": "MinerU 已生成 block，但内容为空",
        "missing_block": "MinerU 完全没有生成 block",
        "visual_region_text": "PDF 文本位于图片/图表区域",
    }
    for kind in ["block_text_repaired", "block_text_difference_observed", "empty_block_recovered", "missing_block", "visual_region_text"]:
        selected = [item for item in findings if item["kind"] == kind]
        lines.extend([
            f"### 4.{list(labels).index(kind) + 1} {labels[kind]}（{len(selected)}项）",
            "",
            "| 页 | Block | 原页证据 / PDF 文本层参照 | MinerU 原结果 | 后处理结果 | 状态/说明 |",
            "|---:|---|---|---|---|---|",
        ])
        for item in selected:
            block_ids = item.get("block_ids") or [item.get("block_id")]
            block_ids = [value for value in block_ids if value]
            block = blocks_by_id.get(block_ids[0]) if block_ids else None
            original = item.get("reference_excerpt") or item.get("source_text") or ""
            if kind == "missing_block":
                mineru = "（未生成 block）"
            elif block is None:
                mineru = "（无法关联当前 block）"
            else:
                mineru = block.raw_content or "（block 内容为空）"
            post = block.resolved_content if block else "（只记录区域，未生成独立结果）"
            note = (
                f"status={block.status}; source={block.source}; flags={block.issue_flags}"
                if block else f"kind={kind}"
            )
            lines.append(
                f"| {item['page_no']} | `{','.join(block_ids)}` | {_page_evidence(work_dir, item['page_no'])}<br>{_md(original)} | {_md(mineru)} | {_md(post)} | {_md(note)} |"
            )
        lines.append("")

    lines.extend([
        "## 5. 全部 VLM 后处理结果",
        "",
        "每一行对应一次 block 级 VLM 终态。参照列同时给出原页图；如果无法由文本层取得内容，还会给出 VLM 实际输入图片。MinerU 原结果和最终 block 内容完整保留，不对长表格做省略。",
        "",
        "| 页 | 类型/Block | PDF 原文或图像证据 | MinerU 原结果 | 后处理结果 | 最终状态 |",
        "|---:|---|---|---|---|---|",
    ])
    for event in sorted(vlm_events, key=lambda value: (value.get("page_no") or 0, value.get("block_id") or "")):
        block_id = event.get("block_id") or ""
        block = blocks_by_id.get(block_id)
        if block is None:
            continue
        original_text = _source_text(block.page_no, block.bbox_json, text_layer)
        reference = _md(original_text) if original_text else _evidence(events, block_id)
        original = f"{_page_evidence(work_dir, block.page_no)}<br>{reference}"
        mineru = _md(block.raw_content, empty="（MinerU 内容为空）")
        post = _md(block.resolved_content, empty="（后处理内容为空）")
        details = event.get("details") or {}
        error = details.get("error_type") or details.get("message")
        status = (
            f"{block.status}; task={event.get('action')}; source={block.source}; "
            f"flags={block.issue_flags}; uncertainty={details.get('uncertainty_count', 0)}"
        )
        if error:
            status += f"; error={error}"
        lines.append(
            f"| {block.page_no} | `{block.block_type}`<br>`{block_id}` | {original} | {mineru} | {post} | {_md(status)} |"
        )
    lines.append("")

    lines.extend([
        "## 6. 复杂表格处理明细",
        "",
        "`table_tiles/` 中只有 Qwen 输入图片；单个 tile 的完整返回没有单独落盘，合并后的候选 HTML 保存在 MySQL `block_revisions` 的 `qwen_table_crosscheck` revision。",
        "",
        "| 页 | Block | 动作 | 结果 | 输入证据 | 详细信息 |",
        "|---:|---|---|---|---|---|",
    ])
    for event in table_events:
        refs = "<br>".join(_link(path, f"输入{index}") for index, path in enumerate(event.get("input_refs") or [], 1))
        lines.append(
            f"| {event.get('page_no') or ''} | `{event.get('block_id') or ''}` | `{event.get('action')}` | `{event.get('decision')}` | {refs or '（无）'} | {_md(event.get('details') or {})} |"
        )
    lines.append("")

    lines.extend([
        "### 6.1 第 115 页两张表的独立数字核对",
        "",
        "| Block | MinerU 数字数 | PDF 文本层数字数 | 数字 multiset | 字符 F1 | 结论 |",
        "|---|---:|---:|---|---:|---|",
    ])
    for block_id in ["a799088f52064c2ba93acdc4d68a8a0b", "fbb688e8f97b449ab8345eb27d7c804d"]:
        stored = blocks_by_id.get(block_id)
        if stored is None:
            continue
        raw_block = _same_raw_block(raw_document, stored)
        if raw_block is None:
            continue
        raw_page = next(page for page in raw_document.pages if page.page_no == stored.page_no)
        reference = _block_reference(raw_block, text_layer.pages.get(stored.page_no, []))
        mineru_numbers = number_tokens(raw_block.content)
        pdf_numbers = number_tokens(reference)
        equal = Counter(mineru_numbers) == Counter(pdf_numbers)
        char_f1 = _char_multiset_f1(
            normalize_text(flatten_content(raw_block.content)),
            normalize_text(reference),
        )
        lines.append(
            f"| `{block_id}` | {len(mineru_numbers)} | {len(pdf_numbers)} | {'一致' if equal else '不一致'} | {char_f1:.6f} | MinerU 原表可用；当前失败来自 VLM 方向/切片，不是原表数字错误 |"
        )
    lines.append("")

    lines.extend([
        "## 7. 全部跨页表判断",
        "",
        "| 当前页 | 前块 → 当前块 | 决策 | 分数 | 语义相似度 | 来源 |",
        "|---:|---|---|---:|---:|---|",
    ])
    for event in cross_page_events:
        details = event.get("details") or {}
        refs = event.get("input_refs") or []
        lines.append(
            f"| {event.get('page_no') or ''} | `{' → '.join(refs)}` | `{event.get('decision')}` | {details.get('score', '')} | {details.get('semantic_similarity', '')} | `{details.get('source', '')}` |"
        )
    lines.append("")

    lines.extend([
        "## 8. 全部未解决项",
        "",
        "以下 block 当前被 Chunk 门阻断，不会作为确定事实写入新 Chunk。",
        "",
        "| 页 | Block | 类型 | MinerU 原结果 | 当前结果 | 状态 | 原因/证据 |",
        "|---:|---|---|---|---|---|---|",
    ])
    for block in unresolved:
        event = next((item for item in reversed(vlm_events) if item.get("block_id") == block.block_id), None)
        details = (event or {}).get("details") or {}
        reason = {
            "flags": block.issue_flags,
            "verification_tags": (block.metadata_json or {}).get("verification_tags") or [],
            "error_type": details.get("error_type"),
            "provider_code": details.get("provider_code"),
            "uncertainty_count": details.get("uncertainty_count"),
        }
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | `{block.block_type}` | {_md(block.raw_content, empty='（空）')} | {_md(block.resolved_content, empty='（空）')} | `{block.status}` | {_md(reason)}<br>{_evidence(events, block.block_id)} |"
        )
    lines.append("")

    lines.extend([
        "## 9. Revision 索引",
        "",
        "下表列出所有发生过多来源修订的 block，便于回到 MySQL 查看完整结构化字段。",
        "",
        "| 页 | Block | 当前来源/状态 | Revision 顺序 |",
        "|---:|---|---|---|",
    ])
    for block in blocks:
        block_revisions = revisions_by_block.get(block.block_id, [])
        if len(block_revisions) <= 1:
            continue
        revision_text = " → ".join(
            f"{revision.source}({revision.prompt_version or '-'},{len(revision.content or '')}字符)"
            for revision in block_revisions
        )
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | `{block.source}/{block.status}` | {_md(revision_text)} |"
        )
    lines.append("")

    lines.extend([
        "## 10. 最终结论",
        "",
        f"1. 页级文字后处理记录了全部 {len(findings)} 条发现项，其中 {finding_counts['missing_block']} 条是 MinerU 完全漏 block，{finding_counts['empty_block_recovered']} 条是空 block，{finding_counts['block_text_difference_observed']} 条已有内容差异按非破坏原则保留 MinerU，历史自动文字修订为 {finding_counts['block_text_repaired']} 条。",
        f"2. VLM 的 {len(vlm_events)} 个 block 中有 {vlm_counts['repaired']} 个完成修复；当前仍有 {len(unresolved)} 个 block 未通过，报告没有把它们伪装成已准确。",
        "3. 第 115 页两张表的 MinerU 原始 HTML 与 PDF 文本层核对通过；当前 `unreadable` 是后处理方向和切片错误，应修正规则后恢复 MinerU 原表或重新正确旋转复核。",
        "4. 第 87 页对象应保持 `chart`，不应因为 Qwen 用 HTML table 表达图表数据就触发 `table_recheck`。",
        f"5. 当前 MySQL/Milvus 已写入 {chunk_count} 个通过门控的 chunk；未解决 block 不在确定事实 Chunk 中，因此表现为召回缺失而不是错误内容泄漏。",
        "",
        "关联文件：",
        "",
        f"- {_link(audit_summary_path, '页面审计汇总')}",
        f"- {_link(audit_path, '页面审计逐条明细')}",
        f"- {_link(workflow_path, '完整工作流 JSONL')}",
        f"- {_link(work_dir / 'runtime_logs' / f'{args.run_id}.log', '运行日志')}",
        f"- {_link(work_dir / 'table_tiles', '表格切片目录')}",
        f"- {_link(work_dir / 'table_recheck', '二次 MinerU 表格复核目录')}",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "line_count": len(lines),
        "page_findings": len(findings),
        "vlm_results": len(vlm_events),
        "table_events": len(table_events),
        "cross_page_events": len(cross_page_events),
        "unresolved": len(unresolved),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
