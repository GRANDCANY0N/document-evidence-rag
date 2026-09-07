from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="milvus_lite",
)
from pymilvus import DataType, MilvusClient

from mineru_vlm_rag.domain.models import Chunk


class MilvusRepository:
    def __init__(self, uri: str, collection_prefix: str = "document_chunks") -> None:
        if "://" not in uri:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri)
        self.collection_prefix = collection_prefix

    def collection_name(self, dimension: int) -> str:
        return f"{self.collection_prefix}_{dimension}"

    def ensure_collection(self, dimension: int) -> str:
        name = self.collection_name(dimension)
        if self.client.has_collection(name):
            return name
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("block_type", DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_role", DataType.VARCHAR, max_length=32)
        schema.add_field("retrieval_tier", DataType.VARCHAR, max_length=16)
        schema.add_field("requires_human_review", DataType.BOOL)
        schema.add_field("document_order", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("section_path", DataType.VARCHAR, max_length=2048)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("asset_id", DataType.VARCHAR, max_length=64)
        schema.add_field("bbox_json", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
        index = self.client.prepare_index_params()
        index.add_index(field_name="dense_vector", index_type="FLAT", metric_type="COSINE")
        self.client.create_collection(name, schema=schema, index_params=index)
        return name

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        replace_documents: bool = True,
    ) -> str:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be non-empty and have equal length")
        dimension = len(vectors[0])
        name = self.ensure_collection(dimension)
        if replace_documents:
            for document_id in sorted({chunk.document_id for chunk in chunks}):
                self.client.delete(name, filter=f'document_id == "{self._escape_filter_value(document_id)}"')
        data = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != dimension:
                raise ValueError("Embedding dimensions are inconsistent")
            data.append({
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "block_type": chunk.block_type,
                "chunk_role": str(chunk.metadata.get("chunk_role") or "unknown")[:32],
                "retrieval_tier": str(chunk.metadata.get("retrieval_tier") or "verified")[:16],
                "requires_human_review": bool(chunk.metadata.get("requires_human_review")),
                "document_order": int(chunk.metadata.get("document_order") or 0),
                "parent_id": chunk.parent_id or "",
                "section_path": chunk.section_path[:2048],
                "text": chunk.display_text[:65535],
                "asset_id": chunk.asset_id or "",
                "bbox_json": chunk.bbox_json[:65535],
                "dense_vector": vector,
            })
        self.client.upsert(name, data=data)
        return name

    @staticmethod
    def _escape_filter_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        document_id: str | None = None,
        block_type: str | None = None,
    ) -> list[dict[str, Any]]:
        name = self.collection_name(len(query_vector))
        if not self.client.has_collection(name):
            return []
        filters = []
        if document_id:
            filters.append(f'document_id == "{self._escape_filter_value(document_id)}"')
        if block_type:
            filters.append(f'block_type == "{self._escape_filter_value(block_type)}"')
        results = self.client.search(
            name,
            data=[query_vector],
            anns_field="dense_vector",
            filter=" and ".join(filters),
            limit=limit,
            output_fields=[
                "document_id", "page_start", "page_end", "block_type", "chunk_role",
                "retrieval_tier", "requires_human_review", "document_order", "parent_id",
                "section_path", "text", "asset_id", "bbox_json",
            ],
        )
        hits: list[dict[str, Any]] = []
        for item in results[0] if results else []:
            entity = dict(item.get("entity") or {})
            entity["id"] = item.get("id")
            entity["distance"] = item.get("distance")
            hits.append(entity)
        return hits

    def list_chunks(
        self,
        dimension: int,
        *,
        document_id: str | None = None,
        block_type: str | None = None,
        limit: int = 16384,
    ) -> list[dict[str, Any]]:
        """Read chunk metadata/text for the small in-process lexical sidecar."""
        name = self.collection_name(dimension)
        if not self.client.has_collection(name):
            return []
        filters = []
        if document_id:
            filters.append(f'document_id == "{self._escape_filter_value(document_id)}"')
        if block_type:
            filters.append(f'block_type == "{self._escape_filter_value(block_type)}"')
        return list(self.client.query(
            name,
            filter=" and ".join(filters) or 'id != ""',
            limit=limit,
            output_fields=[
                "id", "document_id", "page_start", "page_end", "block_type", "chunk_role",
                "retrieval_tier", "requires_human_review", "document_order", "parent_id",
                "section_path", "text", "asset_id", "bbox_json",
            ],
        ))

    def close(self) -> None:
        self.client.close()
