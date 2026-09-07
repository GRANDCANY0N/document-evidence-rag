#!/usr/bin/env python3
"""Retry only REVIEW/UNREADABLE blocks and incrementally add successful chunks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mineru_vlm_rag.chunking import build_chunks
from mineru_vlm_rag.domain.models import (
    Asset,
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
    ResolutionStatus,
)
from mineru_vlm_rag.persistence.mysql_repository import (
    AssetRow,
    BlockRow,
    ChunkRow,
    DocumentRow,
    PageRow,
    RevisionRow,
)
from mineru_vlm_rag.adapters.siliconflow import VLMResult
from mineru_vlm_rag.pipeline.ingest import IngestionPipeline, _append_tags
from mineru_vlm_rag.settings import load_settings
from mineru_vlm_rag.workflow import WorkflowRecorder


LOGGER = logging.getLogger("retry_pending_vlm")


def _json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value


def _load_unit(session: Session, row: BlockRow, document_row: DocumentRow) -> tuple[Document, Page, Block]:
    page_row = session.scalar(
        select(PageRow).where(PageRow.document_id == row.document_id, PageRow.page_no == row.page_no)
    )
    if page_row is None:
        raise RuntimeError(f"page missing for block {row.block_id}")
    block = Block(
        block_id=row.block_id,
        document_id=row.document_id,
        page_no=row.page_no,
        block_type=BlockType(row.block_type),
        bbox=BoundingBox.from_sequence(_json_value(row.bbox_json, None)),
        reading_order=row.reading_order,
        raw_content=row.raw_content,
        resolved_content=row.resolved_content,
        source=row.source,
        asset_id=row.asset_id,
        parent_id=row.parent_id,
        section_path=list(_json_value(row.section_path, [])),
        issue_flags=list(_json_value(row.issue_flags, [])),
        status=ResolutionStatus(row.status),
        metadata=dict(_json_value(row.metadata_json, {})),
    )
    assets: list[Asset] = []
    if row.asset_id:
        asset_row = session.get(AssetRow, row.asset_id)
        if asset_row is None:
            raise RuntimeError(f"asset missing for block {row.block_id}: {row.asset_id}")
        assets.append(
            Asset(
                asset_id=asset_row.asset_id,
                document_id=asset_row.document_id,
                page_no=asset_row.page_no,
                kind=asset_row.kind,
                mime_type=asset_row.mime_type,
                sha256=asset_row.sha256,
                width=asset_row.width,
                height=asset_row.height,
                data=b"",
                source_path=asset_row.source_path,
            )
        )
    page = Page(
        page_id=page_row.page_id,
        document_id=row.document_id,
        page_no=row.page_no,
        width=page_row.width,
        height=page_row.height,
        rotation=page_row.rotation,
        metadata=dict(_json_value(page_row.metadata_json, {})),
        blocks=[block],
    )
    document = Document(
        document_id=document_row.document_id,
        file_name=document_row.file_name,
        source_path=document_row.source_path,
        sha256=document_row.sha256,
        page_count=document_row.page_count,
        status=document_row.status,
        mineru_batch_id=document_row.mineru_batch_id,
        mineru_archive_path=document_row.mineru_archive_path,
        metadata=dict(_json_value(document_row.metadata_json, {})),
        pages=[page],
        assets=assets,
    )
    return document, page, block


def _persist_result(
    session: Session,
    row: BlockRow,
    block: Block,
    chunks: list[Any],
) -> None:
    row.block_type = block.block_type.value
    row.resolved_content = block.resolved_content
    row.source = block.source
    row.asset_id = block.asset_id
    row.issue_flags = block.issue_flags
    row.status = block.status.value
    row.metadata_json = block.metadata
    for revision in block.revisions:
        session.merge(
            RevisionRow(
                revision_id=revision.revision_id,
                block_id=block.block_id,
                source=revision.source,
                content=revision.content,
                structured_data=revision.structured_data,
                model_name=revision.model_name,
                prompt_version=revision.prompt_version,
                created_at=revision.created_at,
            )
        )
    for chunk in chunks:
        session.merge(
            ChunkRow(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                block_type=chunk.block_type,
                parent_id=chunk.parent_id,
                section_path=chunk.section_path,
                display_text=chunk.display_text,
                embedding_text=chunk.embedding_text,
                asset_id=chunk.asset_id,
                bbox_json=chunk.bbox_json,
                metadata_json=chunk.metadata,
                embedding_state="embedded",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--block-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force-task",
        choices=["chart_extract", "ocr_verify", "visual_analyze"],
        help="Use one focused VLM task instead of the normal quality-gate route.",
    )
    parser.add_argument(
        "--recheck-stored-table",
        action="store_true",
        help="Re-evaluate the latest stored Qwen table HTML without another VLM request.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    settings = load_settings()
    pipeline = IngestionPipeline(settings)
    pipeline.initialize()
    engine = create_engine(settings.mysql_dsn, pool_pre_ping=True)
    recorder = WorkflowRecorder(args.document_id, settings.project_root / "outputs" / "workflow_logs")
    work_dir = settings.project_root / "outputs" / args.document_id
    report_path = work_dir / "postprocess" / f"retry_pending_{recorder.run_id}.json"
    results: list[dict[str, Any]] = []
    try:
        with Session(engine) as session:
            document_row = session.get(DocumentRow, args.document_id)
            if document_row is None:
                raise SystemExit(f"document not found: {args.document_id}")
            statement = (
                select(BlockRow)
                .where(BlockRow.document_id == args.document_id)
                .order_by(BlockRow.page_no, BlockRow.reading_order)
            )
            if not args.recheck_stored_table:
                statement = statement.where(BlockRow.status.in_(["review", "unreadable"]))
            if args.block_id:
                statement = statement.where(BlockRow.block_id.in_(args.block_id))
            rows = list(session.scalars(statement))
            if args.limit > 0:
                rows = rows[: args.limit]
            LOGGER.info("pending blocks=%s", len(rows))
            source_pdf = Path(document_row.source_path)
            for index, row in enumerate(rows, start=1):
                old_status = row.status
                document, page, block = _load_unit(session, row, document_row)
                LOGGER.info(
                    "retry start index=%s/%s page=%s block=%s type=%s old_status=%s",
                    index, len(rows), block.page_no, block.block_id, block.block_type.value, old_status,
                )
                if args.recheck_stored_table:
                    if block.block_type != BlockType.TABLE:
                        raise SystemExit("--recheck-stored-table only supports table blocks")
                    stored_revision = session.scalar(
                        select(RevisionRow)
                        .where(
                            RevisionRow.block_id == block.block_id,
                            RevisionRow.source == "qwen_table_crosscheck",
                            RevisionRow.content != "",
                        )
                        .order_by(RevisionRow.created_at.desc())
                    )
                    if stored_revision is None:
                        raise SystemExit(f"stored Qwen table revision missing: {block.block_id}")
                    pipeline._apply_table_crosscheck(
                        block,
                        VLMResult(
                            task="table_to_cells",
                            readable=True,
                            html=stored_revision.content,
                            structured_data={"reused_revision_id": stored_revision.revision_id},
                        ),
                        prompt_version="stored-table-recheck-v1",
                    )
                    recorder.record(
                        "table_recovery",
                        "recheck_stored_mineru_qwen_cells",
                        block.status.value,
                        page_no=block.page_no,
                        block_id=block.block_id,
                        issue_flags=block.issue_flags,
                        details={"reused_revision_id": stored_revision.revision_id},
                    )
                elif args.force_task:
                    image_path = pipeline._ensure_block_image(document, block, source_pdf, work_dir)
                    result = pipeline.silicon.analyze_images(
                        [image_path],
                        args.force_task,
                        context=(
                            f"页码={block.page_no}; block_type={block.block_type.value}; "
                            f"标题={block.metadata.get('chart_caption') or block.metadata.get('caption') or ''}; "
                            "这是失败项的定向复核，只提取图中可见事实，不得猜测。"
                        ),
                    )
                    pipeline._apply_vlm_result(
                        block,
                        result,
                        prompt_version=f"targeted-{args.force_task}-v1",
                    )
                    recorder.record(
                        "vlm",
                        args.force_task,
                        block.status.value,
                        page_no=block.page_no,
                        block_id=block.block_id,
                        issue_flags=block.issue_flags,
                        input_refs=[str(image_path)],
                        details={
                            "targeted_retry": True,
                            "readable": result.readable,
                            "output_length": len(block.content),
                            "uncertainty_count": len(result.uncertainty),
                        },
                    )
                else:
                    pipeline._process_block(document, page, block, source_pdf, work_dir, recorder, use_vlm=True)
                chunks = []
                vectors: list[list[float]] = []
                succeeded = block.status in {ResolutionStatus.ACCEPTED, ResolutionStatus.REPAIRED}
                if succeeded:
                    _append_tags(block, "processing_tags", "targeted_vlm_retry")
                    _append_tags(block, "verification_tags", "retry_succeeded")
                    chunks = build_chunks(
                        document,
                        text_size=int(settings.pipeline["text_chunk_chars"]),
                        overlap=int(settings.pipeline["text_chunk_overlap"]),
                    )
                    if chunks:
                        vectors = pipeline.silicon.embed(
                            [chunk.embedding_text for chunk in chunks],
                            dimensions=int(settings.embedding["dimensions"]),
                        )
                        milvus = pipeline._get_milvus()
                        milvus.upsert(chunks, vectors, replace_documents=False)
                        # Do not keep a Milvus Lite gRPC channel idle while the
                        # next VLM request may wait for several minutes.
                        milvus.close()
                        pipeline.milvus = None
                _persist_result(session, row, block, chunks)
                session.commit()
                recorder.record(
                    "targeted_retry",
                    "retry_pending_block",
                    "repaired" if succeeded else block.status.value,
                    page_no=block.page_no,
                    block_id=block.block_id,
                    issue_flags=block.issue_flags,
                    details={
                        "old_status": old_status,
                        "new_status": block.status.value,
                        "chunk_count": len(chunks),
                        "vector_count": len(vectors),
                    },
                )
                results.append(
                    {
                        "page_no": block.page_no,
                        "block_id": block.block_id,
                        "block_type": block.block_type.value,
                        "old_status": old_status,
                        "new_status": block.status.value,
                        "source": block.source,
                        "content_length": len(block.content),
                        "chunk_count": len(chunks),
                    }
                )
                LOGGER.info("retry done result=%s", results[-1])
        pipeline.mysql.save_workflow_events(recorder.events)
        report = {
            "run_id": recorder.run_id,
            "document_id": args.document_id,
            "attempted": len(results),
            "repaired": sum(item["new_status"] in {"accepted", "repaired"} for item in results),
            "remaining": sum(item["new_status"] in {"review", "unreadable"} for item in results),
            "results": results,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
