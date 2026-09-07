from __future__ import annotations

import re
from difflib import SequenceMatcher

from lxml import etree, html

from mineru_vlm_rag.domain.models import Block, BlockType, Document, Page, ResolutionStatus


def _table_root(content: str):
    root = html.fromstring(content)
    if root.tag == "table":
        return root
    tables = root.xpath(".//table")
    return tables[0] if tables else None


def table_header_text(content: str) -> str:
    try:
        table = _table_root(content)
        if table is None:
            return ""
        header_rows = table.xpath(".//thead/tr")
        if not header_rows:
            rows = table.xpath(".//tr")
            if not rows:
                return ""
            first_cells = rows[0].xpath("./th|./td")
            has_structured_header = bool(rows[0].xpath("./th") or any(
                cell.get("rowspan") or cell.get("colspan") for cell in first_cells
            ))
            header_rows = rows[:2] if has_structured_header else rows[:1]
        cells = []
        for row in header_rows:
            cells.extend(" ".join(cell.itertext()).strip() for cell in row.xpath("./th|./td"))
        return re.sub(r"\s+", " ", " | ".join(value for value in cells if value)).strip()
    except Exception:
        return ""


def _column_count(content: str) -> int:
    try:
        table = _table_root(content)
        if table is None:
            return 0
        rows = table.xpath(".//tr")
        return max((sum(int(cell.get("colspan", "1")) for cell in row.xpath("./th|./td")) for row in rows), default=0)
    except Exception:
        return 0


def continuation_score(previous: Block, current: Block, header_embedding_similarity: float | None = None) -> float:
    previous_end_page = max(previous.metadata.get("source_pages") or [previous.page_no])
    if previous_end_page + 1 != current.page_no:
        return 0.0
    prev_header = table_header_text(previous.content)
    curr_header = table_header_text(current.content)
    lexical = SequenceMatcher(None, prev_header, curr_header).ratio() if prev_header and curr_header else 0.0
    columns_a = _column_count(previous.content)
    columns_b = _column_count(current.content)
    column_match = 1.0 if columns_a and columns_a == columns_b else 0.0
    position = 0.0
    if previous.bbox and current.bbox:
        position = 1.0 if previous.bbox.y0 > current.bbox.y0 else 0.5
    keyword = 1.0 if re.search(r"续表|continued", current.content, flags=re.I) else 0.0
    semantic = header_embedding_similarity if header_embedding_similarity is not None else lexical
    return round(0.2 * position + 0.25 * column_match + 0.2 * lexical + 0.25 * semantic + 0.1 * keyword, 4)


_TABLE_REF_RE = re.compile(r"(?:附表|表)\s*([一二三四五六七八九十百\d]+(?:[-—.]\d+)?)", re.I)
_CONTEXT_EXCLUDE_RE = re.compile(
    r"^[（(]?\s*(?:单位|币种|注|说明|数据来源|资料来源)[：:]|^[（(]?\s*(?:截至)?\d{4}年.*[）)]?$",
    re.I,
)


def _normalized_identity(value: str) -> str:
    value = re.sub(r"\s+", "", value or "").lower()
    value = re.sub(r"[（(].*?(?:余额|日期).*?[）)]", "", value)
    return value[:160]


def _coordinate_height(page: Page) -> float:
    value = page.metadata.get("mineru_coordinate_height") or page.height or 1000.0
    return float(value)


def table_identity(page: Page, table: Block) -> dict[str, object]:
    """Collect a table number/title from nearby evidence, excluding units/notes."""
    contexts: list[Block] = []
    for block in page.blocks:
        if block.block_id == table.block_id or not block.content.strip():
            continue
        if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
            continue
        associated = str(block.metadata.get("associated_table_block_id") or "")
        if associated == table.block_id:
            contexts.append(block)
            continue
        if not block.bbox or not table.bbox:
            continue
        gap = table.bbox.y0 - block.bbox.y1
        horizontal_overlap = max(0.0, min(block.bbox.x1, table.bbox.x1) - max(block.bbox.x0, table.bbox.x0))
        if 0 <= gap <= _coordinate_height(page) * 0.09 and horizontal_overlap > 0:
            contexts.append(block)

    values = [
        block.content.strip()
        for block in contexts
        if not _CONTEXT_EXCLUDE_RE.match(block.content.strip())
    ]
    references: list[str] = []
    for value in values:
        match = _TABLE_REF_RE.search(value)
        if match:
            references.append(match.group(1))
    titles = [
        value for value in values
        if not re.fullmatch(r"(?:续)?(?:附表|表)\s*[一二三四五六七八九十百\d]+(?:[-—.]\d+)?", re.sub(r"\s+", "", value))
        and not re.fullmatch(r"续表", value.strip(), re.I)
    ]
    return {
        "references": list(dict.fromkeys(references)),
        "titles": list(dict.fromkeys(titles)),
        "text": " ".join(values),
    }


def table_continuation_evidence(
    previous: Block,
    current: Block,
    previous_page: Page,
    current_page: Page,
) -> dict[str, object]:
    """Return hard identity/edge gates that header similarity cannot override."""
    previous_identity = table_identity(previous_page, previous)
    current_identity = table_identity(current_page, current)
    previous_refs = set(str(value) for value in previous_identity["references"])
    current_refs = set(str(value) for value in current_identity["references"])
    distinct_reference = bool(previous_refs and current_refs and previous_refs.isdisjoint(current_refs))

    previous_title = _normalized_identity(" ".join(previous_identity["titles"]))
    current_title = _normalized_identity(" ".join(current_identity["titles"]))
    title_similarity = (
        SequenceMatcher(None, previous_title, current_title).ratio()
        if previous_title and current_title
        else None
    )
    explicit_continuation = bool(re.search(
        r"续表|continued",
        f"{current_identity['text']} {current.content[:300]}",
        re.I,
    ))
    distinct_title = bool(
        previous_title
        and current_title
        and title_similarity is not None
        and title_similarity < 0.72
    )
    independent_current_table = bool(current_refs or current_title) and not explicit_continuation
    hard_negative = distinct_reference or distinct_title

    previous_bottom_ratio = previous.bbox.y1 / _coordinate_height(previous_page) if previous.bbox else 0.0
    current_top_ratio = current.bbox.y0 / _coordinate_height(current_page) if current.bbox else 1.0
    edge_continuity = previous_bottom_ratio >= 0.80 and current_top_ratio <= 0.25
    same_identity = bool(
        (previous_refs and previous_refs == current_refs)
        or (title_similarity is not None and title_similarity >= 0.90)
    )
    auto_eligible = bool(
        edge_continuity
        and not hard_negative
        and (explicit_continuation or same_identity or not independent_current_table)
    )
    return {
        "previous_identity": previous_identity,
        "current_identity": current_identity,
        "distinct_reference": distinct_reference,
        "distinct_title": distinct_title,
        "title_similarity": title_similarity,
        "explicit_continuation": explicit_continuation,
        "edge_continuity": edge_continuity,
        "previous_bottom_ratio": round(previous_bottom_ratio, 4),
        "current_top_ratio": round(current_top_ratio, 4),
        "hard_negative": hard_negative,
        "auto_eligible": auto_eligible,
    }


def revert_unsafe_table_merges(document: Document) -> dict[str, object]:
    """Undo old destructive merge chains when adjacent pages have new identities."""
    pages = {page.page_no: page for page in document.pages}
    blocks = {block.block_id: block for page in document.pages for block in page.blocks}
    children_by_parent: dict[str, list[Block]] = {}
    for block in blocks.values():
        parent_id = str(block.metadata.get("merged_into") or "")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(block)

    reverted_parents: list[str] = []
    reverted_children: list[str] = []
    for parent_id, children in children_by_parent.items():
        parent = blocks.get(parent_id)
        if parent is None or not parent.raw_content.strip():
            continue
        unsafe = False
        reasons: list[dict[str, object]] = []
        for child in sorted(children, key=lambda value: value.page_no):
            previous_page = pages.get(child.page_no - 1) or pages.get(parent.page_no)
            current_page = pages.get(child.page_no)
            if previous_page is None or current_page is None:
                continue
            evidence = table_continuation_evidence(parent, child, previous_page, current_page)
            if evidence["hard_negative"]:
                unsafe = True
                reasons.append({"page_no": child.page_no, **evidence})
        if not unsafe:
            continue

        merged_pages = list(parent.metadata.get("source_pages") or [parent.page_no])
        parent.resolved_content = parent.raw_content
        parent.metadata.pop("source_pages", None)
        parent.metadata["reverted_merged_source_pages"] = merged_pages
        parent.metadata["table_merge_reverted"] = True
        parent.metadata["table_merge_revert_reasons"] = reasons
        parent.issue_flags = [flag for flag in parent.issue_flags if flag != "TAB-003"]
        previous_status = str(parent.metadata.pop("pre_merge_status", "") or "")
        try:
            parent.status = ResolutionStatus(previous_status) if previous_status else ResolutionStatus.ACCEPTED
        except ValueError:
            parent.status = ResolutionStatus.ACCEPTED
        parent.metadata.setdefault("verification_tags", []).append("unsafe_table_merge_reverted")
        reverted_parents.append(parent.block_id)

        for child in children:
            child.metadata.pop("skip_chunk", None)
            child.metadata.pop("merged_into", None)
            child.metadata.setdefault("verification_tags", []).append("unsafe_table_merge_reverted")
            reverted_children.append(child.block_id)

    return {
        "reverted_parent_count": len(reverted_parents),
        "reverted_child_count": len(reverted_children),
        "reverted_parent_ids": reverted_parents,
        "reverted_child_ids": reverted_children,
    }


def merge_table_html(
    previous_html: str,
    current_html: str,
    previous_page_no: int | None = None,
    current_page_no: int | None = None,
) -> str:
    previous = _table_root(previous_html)
    current = _table_root(current_html)
    if previous is None or current is None:
        raise ValueError("Both table fragments must contain valid table HTML")
    previous_body = previous.xpath("./tbody")
    if previous_body:
        target = previous_body[0]
    else:
        direct_rows = previous.xpath("./tr")
        target = etree.SubElement(previous, "tbody")
        if direct_rows:
            if previous.xpath("./thead"):
                body_rows = direct_rows
            else:
                first_cells = direct_rows[0].xpath("./th|./td")
                has_structured_header = bool(direct_rows[0].xpath("./th") or any(
                    cell.get("rowspan") or cell.get("colspan") for cell in first_cells
                ))
                header_count = min(len(direct_rows), 2 if has_structured_header else 1)
                head = etree.Element("thead")
                previous.insert(0, head)
                for row in direct_rows[:header_count]:
                    head.append(row)
                body_rows = direct_rows[header_count:]
            for row in body_rows:
                target.append(row)
    if previous_page_no is not None:
        for row in previous.xpath("./tbody/tr"):
            if not row.get("data-source-page"):
                row.set("data-source-page", str(previous_page_no))
    current_rows = current.xpath("./tbody/tr") or current.xpath("./tr")
    header_text = table_header_text(previous_html)
    for row in current_rows:
        row_text = re.sub(r"\s+", " ", " ".join(row.itertext())).strip()
        if header_text and SequenceMatcher(None, header_text, row_text).ratio() > 0.75:
            continue
        if current_page_no is not None:
            row.set("data-source-page", str(current_page_no))
        target.append(row)
    return etree.tostring(previous, encoding="unicode", method="html")
