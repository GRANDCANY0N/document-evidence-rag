from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BlockType(StrEnum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FORMULA = "formula"
    SEAL = "seal"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    UNKNOWN = "unknown"


class ResolutionStatus(StrEnum):
    ACCEPTED = "accepted"
    REPAIRED = "repaired"
    REVIEW = "review"
    UNREADABLE = "unreadable"
    FAILED = "failed"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_sequence(cls, value: list[float] | tuple[float, ...] | None) -> "BoundingBox | None":
        if not value or len(value) < 4:
            return None
        return cls(x0=float(value[0]), y0=float(value[1]), x1=float(value[2]), y1=float(value[3]))

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


class Revision(BaseModel):
    revision_id: str = Field(default_factory=new_id)
    source: str
    content: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None
    prompt_version: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Asset(BaseModel):
    asset_id: str = Field(default_factory=new_id)
    document_id: str
    page_no: int
    kind: str
    mime_type: str
    sha256: str
    width: int | None = None
    height: int | None = None
    data: bytes = b""
    source_path: str | None = None


class Block(BaseModel):
    block_id: str = Field(default_factory=new_id)
    document_id: str
    page_no: int
    block_type: BlockType = BlockType.UNKNOWN
    bbox: BoundingBox | None = None
    reading_order: int = 0
    raw_content: str = ""
    resolved_content: str = ""
    source: str = "mineru"
    asset_id: str | None = None
    parent_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    issue_flags: list[str] = Field(default_factory=list)
    status: ResolutionStatus = ResolutionStatus.ACCEPTED
    metadata: dict[str, Any] = Field(default_factory=dict)
    revisions: list[Revision] = Field(default_factory=list)

    @property
    def content(self) -> str:
        return self.resolved_content or self.raw_content


class Page(BaseModel):
    page_id: str = Field(default_factory=new_id)
    document_id: str
    page_no: int
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    blocks: list[Block] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: str
    file_name: str
    source_path: str
    sha256: str
    page_count: int
    status: str = "pending"
    mineru_batch_id: str | None = None
    mineru_archive_path: str | None = None
    pages: list[Page] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=new_id)
    document_id: str
    page_start: int
    page_end: int
    block_type: str
    parent_id: str | None = None
    section_path: str = ""
    display_text: str
    embedding_text: str
    asset_id: str | None = None
    bbox_json: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvent(BaseModel):
    event_id: str = Field(default_factory=new_id)
    document_id: str
    stage: str
    action: str
    decision: str
    page_no: int | None = None
    block_id: str | None = None
    issue_flags: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
