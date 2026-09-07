from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from mineru_vlm_rag.domain.models import (
    Block,
    BlockType,
    BoundingBox,
    Chunk,
    Document,
    Page,
    ResolutionStatus,
    WorkflowEvent,
    utcnow,
)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(512))
    source_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    page_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), index=True)
    mineru_batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mineru_archive_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class PageRow(Base):
    __tablename__ = "pages"
    page_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    width: Mapped[float | None]
    height: Mapped[float | None]
    rotation: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("document_id", "page_no", name="uq_document_page"),)


class AssetRow(Base):
    __tablename__ = "assets"
    asset_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    mime_type: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[bytes] = mapped_column(LargeBinary(length=2**32 - 1))
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class BlockRow(Base):
    __tablename__ = "blocks"
    block_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer, index=True)
    block_type: Mapped[str] = mapped_column(String(32), index=True)
    bbox_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reading_order: Mapped[int] = mapped_column(Integer)
    raw_content: Mapped[str] = mapped_column(Text)
    resolved_content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32))
    asset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    section_path: Mapped[list] = mapped_column(JSON, default=list)
    issue_flags: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class RevisionRow(Base):
    __tablename__ = "block_revisions"
    revision_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    block_id: Mapped[str] = mapped_column(ForeignKey("blocks.block_id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    structured_data: Mapped[dict] = mapped_column(JSON, default=dict)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    block_type: Mapped[str] = mapped_column(String(32), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    section_path: Mapped[str] = mapped_column(Text)
    display_text: Mapped[str] = mapped_column(Text)
    embedding_text: Mapped[str] = mapped_column(Text)
    asset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bbox_json: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding_state: Mapped[str] = mapped_column(String(32), default="pending", index=True)


class WorkflowEventRow(Base):
    __tablename__ = "workflow_events"
    event_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(64), index=True)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    issue_flags: Mapped[list] = mapped_column(JSON, default=list)
    input_refs: Mapped[list] = mapped_column(JSON, default=list)
    output_refs: Mapped[list] = mapped_column(JSON, default=list)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MySQLRepository:
    def __init__(self, dsn: str) -> None:
        self.engine = create_engine(dsn, pool_pre_ping=True)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def load_document_graph(self, document_id: str) -> Document:
        """Load the resolved page/block graph needed for an offline rechunk."""
        with Session(self.engine) as session:
            document_row = session.get(DocumentRow, document_id)
            if document_row is None:
                raise KeyError(f"Document not found: {document_id}")
            page_rows = list(session.scalars(
                select(PageRow)
                .where(PageRow.document_id == document_id)
                .order_by(PageRow.page_no)
            ))
            block_rows = list(session.scalars(
                select(BlockRow)
                .where(BlockRow.document_id == document_id)
                .order_by(BlockRow.page_no, BlockRow.reading_order)
            ))

        pages_by_no = {
            row.page_no: Page(
                page_id=row.page_id,
                document_id=row.document_id,
                page_no=row.page_no,
                width=row.width,
                height=row.height,
                rotation=row.rotation,
                metadata=row.metadata_json or {},
            )
            for row in page_rows
        }
        for row in block_rows:
            try:
                block_type = BlockType(row.block_type)
            except ValueError:
                block_type = BlockType.UNKNOWN
            try:
                status = ResolutionStatus(row.status)
            except ValueError:
                status = ResolutionStatus.REVIEW
            page = pages_by_no.setdefault(
                row.page_no,
                Page(document_id=document_id, page_no=row.page_no),
            )
            page.blocks.append(
                Block(
                    block_id=row.block_id,
                    document_id=row.document_id,
                    page_no=row.page_no,
                    block_type=block_type,
                    bbox=BoundingBox.from_sequence(row.bbox_json),
                    reading_order=row.reading_order,
                    raw_content=row.raw_content or "",
                    resolved_content=row.resolved_content or "",
                    source=row.source,
                    asset_id=row.asset_id,
                    parent_id=row.parent_id,
                    section_path=row.section_path or [],
                    issue_flags=row.issue_flags or [],
                    status=status,
                    metadata=row.metadata_json or {},
                )
            )
        return Document(
            document_id=document_row.document_id,
            file_name=document_row.file_name,
            source_path=document_row.source_path,
            sha256=document_row.sha256,
            page_count=document_row.page_count,
            status=document_row.status,
            mineru_batch_id=document_row.mineru_batch_id,
            mineru_archive_path=document_row.mineru_archive_path,
            pages=sorted(pages_by_no.values(), key=lambda page: page.page_no),
            metadata=document_row.metadata_json or {},
            created_at=document_row.created_at,
        )

    def save_document_graph(self, document: Document, chunks: list[Chunk]) -> None:
        with Session(self.engine) as session, session.begin():
            session.merge(
                DocumentRow(
                    document_id=document.document_id,
                    file_name=document.file_name,
                    source_path=document.source_path,
                    sha256=document.sha256,
                    page_count=document.page_count,
                    status=document.status,
                    mineru_batch_id=document.mineru_batch_id,
                    mineru_archive_path=document.mineru_archive_path,
                    metadata_json=document.metadata,
                    created_at=document.created_at,
                    updated_at=utcnow(),
                )
            )
            session.execute(delete(RevisionRow).where(RevisionRow.block_id.in_(select(BlockRow.block_id).where(BlockRow.document_id == document.document_id))))
            session.execute(delete(BlockRow).where(BlockRow.document_id == document.document_id))
            session.execute(delete(AssetRow).where(AssetRow.document_id == document.document_id))
            session.execute(delete(PageRow).where(PageRow.document_id == document.document_id))
            session.execute(delete(ChunkRow).where(ChunkRow.document_id == document.document_id))
            for page in document.pages:
                session.add(PageRow(
                    page_id=page.page_id, document_id=document.document_id, page_no=page.page_no,
                    width=page.width, height=page.height, rotation=page.rotation, metadata_json=page.metadata,
                ))
            for asset in document.assets:
                session.add(AssetRow(
                    asset_id=asset.asset_id, document_id=document.document_id, page_no=asset.page_no,
                    kind=asset.kind, mime_type=asset.mime_type, sha256=asset.sha256,
                    width=asset.width, height=asset.height, payload=asset.data, source_path=asset.source_path,
                ))
            for page in document.pages:
                for block in page.blocks:
                    session.add(BlockRow(
                        block_id=block.block_id, document_id=document.document_id, page_no=block.page_no,
                        block_type=block.block_type.value, bbox_json=block.bbox.as_list() if block.bbox else None,
                        reading_order=block.reading_order, raw_content=block.raw_content,
                        resolved_content=block.resolved_content, source=block.source, asset_id=block.asset_id,
                        parent_id=block.parent_id, section_path=block.section_path,
                        issue_flags=block.issue_flags, status=block.status.value, metadata_json=block.metadata,
                    ))
                    for revision in block.revisions:
                        session.add(RevisionRow(
                            revision_id=revision.revision_id, block_id=block.block_id, source=revision.source,
                            content=revision.content, structured_data=revision.structured_data,
                            model_name=revision.model_name, prompt_version=revision.prompt_version,
                            created_at=revision.created_at,
                        ))
            for chunk in chunks:
                session.add(ChunkRow(
                    chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                    page_start=chunk.page_start, page_end=chunk.page_end, block_type=chunk.block_type,
                    parent_id=chunk.parent_id, section_path=chunk.section_path,
                    display_text=chunk.display_text, embedding_text=chunk.embedding_text,
                    asset_id=chunk.asset_id, bbox_json=chunk.bbox_json,
                    metadata_json=chunk.metadata, embedding_state="pending",
                ))

    def mark_chunks_embedded(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        with Session(self.engine) as session, session.begin():
            rows = session.scalars(select(ChunkRow).where(ChunkRow.chunk_id.in_(ids))).all()
            for row in rows:
                row.embedding_state = "embedded"

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Iterable[Chunk],
        *,
        embedding_state: str = "embedded",
    ) -> int:
        """Atomically replace only a document's chunk rows.

        Pages, blocks, revisions, assets and workflow events are deliberately
        untouched.  This is the safe persistence path for offline rechunking
        followed by a separately completed Milvus upsert.
        """
        values = list(chunks)
        if any(chunk.document_id != document_id for chunk in values):
            raise ValueError("All chunks must belong to the requested document")
        chunk_ids = [chunk.chunk_id for chunk in values]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk IDs must be unique")
        with Session(self.engine) as session, session.begin():
            if session.get(DocumentRow, document_id) is None:
                raise KeyError(f"Document not found: {document_id}")
            session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
            for chunk in values:
                session.add(ChunkRow(
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
                    embedding_state=embedding_state,
                ))
        return len(values)

    def update_document_status(self, document_id: str, status: str) -> None:
        with Session(self.engine) as session, session.begin():
            session.execute(
                update(DocumentRow)
                .where(DocumentRow.document_id == document_id)
                .values(status=status, updated_at=utcnow())
            )

    def save_workflow_events(self, events: Iterable[WorkflowEvent]) -> None:
        rows = list(events)
        if not rows:
            return
        with Session(self.engine) as session, session.begin():
            for event in rows:
                session.merge(WorkflowEventRow(
                    event_id=event.event_id, document_id=event.document_id,
                    stage=event.stage, action=event.action, decision=event.decision,
                    page_no=event.page_no, block_id=event.block_id,
                    issue_flags=event.issue_flags, input_refs=event.input_refs,
                    output_refs=event.output_refs, details_json=event.details,
                    created_at=event.created_at,
                ))

    def fetch_asset(self, asset_id: str) -> tuple[str, bytes] | None:
        with Session(self.engine) as session:
            row = session.get(AssetRow, asset_id)
            return (row.mime_type, row.payload) if row else None
