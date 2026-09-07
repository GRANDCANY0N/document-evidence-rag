from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mineru_vlm_rag.persistence.mysql_repository import BlockRow, MySQLRepository
from mineru_vlm_rag.settings import load_settings


FOOTNOTE_MAP = {"①": "a", "②": "b", "③": "c", "④": "d", "⑤": "e"}
SENTENCE_END = ("。", "！", "？", "；")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report text-layer post-processing differences")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _md(value: Any, *, empty: str = "（空）") -> str:
    if value is None or value == "":
        return empty
    text = html.escape(str(value), quote=False).replace("|", "&#124;")
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def _link(path: Path, label: str) -> str:
    return f"[{label}](<{path}>)"


def _page_link(work_dir: Path, page_no: int) -> str:
    return _link(work_dir / "pages" / f"page-{page_no:04d}.png", "原页图")


def _normalized(value: str | None) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def _canonical(value: str | None) -> str:
    value = _normalized(value)
    return re.sub(r"[$\\{}^]", "", value)


def _footnote_pairs(raw: str, resolved: str) -> list[str]:
    pairs: list[str] = []
    for symbol, letter in FOOTNOTE_MAP.items():
        count = raw.count(symbol)
        if count and re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", resolved):
            pairs.append(f"{symbol}→{letter}（{count}处）")
    return pairs


def _repeated_prefix(raw: str, resolved: str) -> str:
    source = _normalized(raw)
    target = _normalized(resolved)
    candidates = [source[:size] for size in range(8, min(40, len(source)) + 1) if target.count(source[:size]) > 1]
    return candidates[-1] if candidates else ""


def _duplicate_insertions(raw: str, resolved: str) -> list[str]:
    source = _normalized(raw)
    target = _normalized(resolved)
    inserted: list[str] = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
        segment = target[j1:j2]
        if tag == "insert" and len(segment) >= 2 and target.count(segment) > source.count(segment):
            inserted.append(segment)
    return list(dict.fromkeys(inserted))


def _strong_truncation(raw: str, resolved: str) -> bool:
    source = _canonical(raw)
    target = _canonical(resolved)
    return (
        raw.strip().endswith(SENTENCE_END)
        and not resolved.strip().endswith(SENTENCE_END)
        and len(source) > len(target) + 5
    )


def _diff_segments(raw: str, resolved: str) -> tuple[str, str]:
    source = _normalized(raw)
    target = _normalized(resolved)
    only_source: list[str] = []
    only_target: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
        if tag in {"delete", "replace"} and i1 != i2:
            only_source.append(source[i1:i2])
        if tag in {"insert", "replace"} and j1 != j2:
            only_target.append(target[j1:j2])
    return " … ".join(only_source), " … ".join(only_target)


def _excerpt(value: str, marker: str, radius: int = 45) -> str:
    compact = " ".join((value or "").split())
    position = compact.find(marker)
    if position < 0:
        return compact[: radius * 2]
    return compact[max(0, position - radius): position + len(marker) + radius]


def main() -> None:
    args = _args()
    settings = load_settings()
    work_dir = settings.project_root / "outputs" / args.document_id
    audit_path = work_dir / "postprocess" / "page_audit.jsonl"
    findings = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    findings = [item for item in findings if item.get("kind") == "block_text_repaired"]
    finding_by_id = {item["block_id"]: item for item in findings}

    repository = MySQLRepository(settings.mysql_dsn)
    with Session(repository.engine) as session:
        blocks = list(session.scalars(
            select(BlockRow)
            .where(BlockRow.block_id.in_(list(finding_by_id)))
            .order_by(BlockRow.page_no, BlockRow.reading_order)
        ))
    repository.engine.dispose()

    categories: dict[str, list[BlockRow]] = {
        "footnote": [],
        "truncation": [],
        "duplicate": [],
        "reorder": [],
        "numeric_other": [],
        "other": [],
    }
    tags_by_id: dict[str, list[str]] = {}
    footnote_occurrences = 0
    for block in blocks:
        raw = block.raw_content or ""
        resolved = block.resolved_content or ""
        finding = finding_by_id[block.block_id]
        tags: list[str] = []
        pairs = _footnote_pairs(raw, resolved)
        if pairs:
            tags.append("脚注编码被改错")
            categories["footnote"].append(block)
            footnote_occurrences += sum(int(re.search(r"（(\d+)处）", pair).group(1)) for pair in pairs)
        if _strong_truncation(raw, resolved):
            tags.append("后处理截断")
            categories["truncation"].append(block)
        repeated = _duplicate_insertions(raw, resolved)
        if repeated and finding.get("character_f1") != 1.0:
            tags.append("后处理重复文字")
            categories["duplicate"].append(block)
        if finding.get("character_f1") == 1.0 and float(finding.get("sequence_similarity") or 1.0) < 0.98:
            tags.append("阅读顺序被改变")
            categories["reorder"].append(block)
        if finding.get("numeric_match") is False and not pairs:
            tags.append("其他数字/符号差异")
            categories["numeric_other"].append(block)
        if not tags:
            tags.append("其他文本层替换（未证明更准确）")
            categories["other"].append(block)
        tags_by_id[block.block_id] = tags

    unique_high_risk = {
        block.block_id
        for key in ("footnote", "truncation", "duplicate", "reorder", "numeric_other")
        for block in categories[key]
    }
    output = args.output or work_dir / "TEXT_POSTPROCESS_DIFF_REPORT.md"
    source_file = settings.project_root / "src" / "mineru_vlm_rag" / "quality" / "completeness.py"

    lines: list[str] = [
        "# MinerU 与文本后处理差异专项报告",
        "",
        f"文档 ID：`{args.document_id}`  ",
        f"原 PDF：`{settings.project_root / 'input' / '2025金融稳定报告.pdf'}`",
        "",
        "## 1. 结论",
        "",
        f"- 本次共有 {len(blocks)} 个已有文字 block 被 PDF 文本层覆盖。",
        f"- 至少 {len(unique_high_risk)} 个 block 触发了可确定或高风险的差异规则；各类型可重叠。",
        f"- 脚注编码改错：{len(categories['footnote'])} 个 block，{footnote_occurrences} 处 ①/②/③/④ 被替换成 a/b/c/d。",
        f"- 明显截断：{len(categories['truncation'])} 个 block。原 MinerU 以句号结束，后处理却在半句中结束。",
        f"- 重复内容：{len(categories['duplicate'])} 个 block。",
        f"- 同一字符集合但阅读顺序被改变：{len(categories['reorder'])} 个 block。",
        f"- 除脚注外仍存在数字/符号差异：{len(categories['numeric_other'])} 个 block。",
        "",
        "PDF 文本层只是参照源，不是原文真值。本报告以“原页图”作为原文证据，并同时列出 MinerU、PDF 文本层参照和最终后处理结果。",
        "",
        "## 2. 第 56 页指定 Block 的完整分析",
        "",
    ]

    target = next(block for block in blocks if block.block_id == "15122bacf08347e8a64c9353cb309f47")
    target_finding = finding_by_id[target.block_id]
    lines.extend([
        f"原页：{_page_link(work_dir, 56)}  ",
        f"MinerU bbox：`{target.bbox_json}`  ",
        f"触发值：`sequence={target_finding['sequence_similarity']}`，`character_f1={target_finding['character_f1']}`，`length_ratio={target_finding['length_ratio']}`，`numeric_match={target_finding['numeric_match']}`。",
        "",
        "| 版本 | 内容 |",
        "|---|---|",
        "| 原页正确结尾 | ……转贴现利率为1.38%，同比下降21个基点；质押式回购利率为1.79%，同比下降1个基点。 |",
        f"| MinerU | {_md(target.raw_content)} |",
        f"| PDF 文本层 bbox 参照 | {_md(target_finding['reference_excerpt'])} |",
        f"| 后处理 | {_md(target.resolved_content)} |",
        "",
        "MinerU 正确把左栏底部与右栏顶部的续行合成完整段落，但 bbox 仍只是左栏的 `[110, 578, 473, 883]`。后处理只选中 bbox 内的文本层行，因此看不到右栏顶部的“率为1.79%，同比下降1个基点。”。",
        "",
        "随后程序只要字符 F1≥0.90 就进入覆盖分支，直接执行 `resolved_content = reference`。虽然该 block 的 `numeric_match=false`，但数字差异的阻断逻辑放在后面的 `elif`，已经无法阻止高 F1 分支。",
        "",
        f"相关源码：{_link(source_file, 'completeness.py')}，关键位置为 `_covers()`、`_block_reference()` 和 `_audit_existing_blocks()`。",
        "",
        "## 3. 全部脚注编码改错",
        "",
        "原页视觉符号是 ①/②/③/④，MinerU 保留了这些符号；`pdftotext -bbox-layout` 读取 PDF 字体的 ToUnicode 映射后得到 a/b/c/d，后处理又用文本层覆盖 MinerU，所以最终结果反而错了。",
        "",
        "| 页 | Block | 原文证据 | 映射 | MinerU 片段 | 后处理片段 |",
        "|---:|---|---|---|---|---|",
    ])
    for block in categories["footnote"]:
        raw = block.raw_content or ""
        resolved = block.resolved_content or ""
        pairs = _footnote_pairs(raw, resolved)
        symbol = next(value for value in FOOTNOTE_MAP if value in raw)
        letter = FOOTNOTE_MAP[symbol]
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | {_page_link(work_dir, block.page_no)} | {_md(', '.join(pairs))} | "
            f"{_md(_excerpt(raw, symbol))} | {_md(_excerpt(resolved, letter))} |"
        )

    lines.extend([
        "",
        "## 4. 明显截断的全部 Block",
        "",
        "| 页 | Block | 原文证据 | MinerU 结尾 | 后处理结尾 | MinerU 独有内容 |",
        "|---:|---|---|---|---|---|",
    ])
    for block in categories["truncation"]:
        only_raw, _ = _diff_segments(block.raw_content or "", block.resolved_content or "")
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | {_page_link(work_dir, block.page_no)} | "
            f"{_md((block.raw_content or '')[-120:])} | {_md((block.resolved_content or '')[-120:])} | {_md(only_raw)} |"
        )

    lines.extend([
        "",
        "## 5. 后处理引入重复内容的全部 Block",
        "",
        "| 页 | Block | 原文证据 | 被重复的文字 | MinerU | 后处理 |",
        "|---:|---|---|---|---|---|",
    ])
    for block in categories["duplicate"]:
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | {_page_link(work_dir, block.page_no)} | "
            f"{_md('；'.join(_duplicate_insertions(block.raw_content or '', block.resolved_content or '')))} | "
            f"{_md(block.raw_content)} | {_md(block.resolved_content)} |"
        )

    lines.extend([
        "",
        "## 6. 全部 80 条自动覆盖明细",
        "",
        "本节不预设后处理一定正确。“原文”通过原页图查看；PDF 文本层参照是当时实际覆盖 MinerU 的内容。",
        "",
        "| 页 | Block | 原文证据 | MinerU | PDF 文本层 / 后处理 | MinerU 独有 | 后处理新增 | 分类 |",
        "|---:|---|---|---|---|---|---|---|",
    ])
    for block in blocks:
        finding = finding_by_id[block.block_id]
        only_raw, only_resolved = _diff_segments(block.raw_content or "", block.resolved_content or "")
        lines.append(
            f"| {block.page_no} | `{block.block_id}` | {_page_link(work_dir, block.page_no)} | "
            f"{_md(block.raw_content)} | {_md(block.resolved_content)} | {_md(only_raw)} | {_md(only_resolved)} | "
            f"{_md('；'.join(tags_by_id[block.block_id]))}<br>"
            f"`seq={finding.get('sequence_similarity')}; f1={finding.get('character_f1')}; numeric={finding.get('numeric_match')}` |"
        )

    lines.extend([
        "",
        "## 7. 根因归纳",
        "",
        "1. **参照源层面**：PDF 文本层的字形和 Unicode 可以不一致，本 PDF 的①—④即是明确例子。",
        "2. **坐标层面**：MinerU 的逻辑 block 可能已合并跨栏续文，但 bbox 仍只覆盖第一段物理区域，用单 bbox 反查文本层会截断。",
        "3. **阅读顺序层面**：`pdftotext` 的 line order 不等于视觉阅读顺序，目录、双栏和加粗行容易错序或重复。",
        "4. **门控层面**：当前实现把 `character_f1 >= 0.90` 当成足够的覆盖条件，没有要求数字一致、结尾完整、无重复、无脚注编码冲突。",
        "",
        "因此，这 80 条不应继续统一标记为“已修复”。本报告只做结果审计，没有改写 MySQL 中的当前内容。",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "automatic_replacements": len(blocks),
        "unique_high_risk": len(unique_high_risk),
        "footnote_blocks": len(categories["footnote"]),
        "footnote_occurrences": footnote_occurrences,
        "strong_truncations": len(categories["truncation"]),
        "duplicates": len(categories["duplicate"]),
        "reorders": len(categories["reorder"]),
        "numeric_other": len(categories["numeric_other"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
