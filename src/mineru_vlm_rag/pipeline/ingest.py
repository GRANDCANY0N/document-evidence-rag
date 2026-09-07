from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from mineru_vlm_rag.adapters.mineru_client import MinerUClient
from mineru_vlm_rag.adapters.siliconflow import SiliconFlowClient, VLMResult
from mineru_vlm_rag.chunking import build_chunks
from mineru_vlm_rag.domain.models import (
    Asset,
    Block,
    BlockType,
    Document,
    ResolutionStatus,
    Revision,
    BoundingBox,
)
from mineru_vlm_rag.normalization import (
    associate_visual_context,
    mark_repeated_templates,
    normalize_document_structure,
    parse_mineru_output,
)
from mineru_vlm_rag.normalization.reading_order import rebuild_reading_order
from mineru_vlm_rag.pdf import create_fresh_submission_copy, enhance_image, inspect_pdf, render_block, render_page
from mineru_vlm_rag.persistence import MilvusRepository, MySQLRepository
from mineru_vlm_rag.quality import (
    DetectionContext,
    detect_block_issues,
    detect_solid_occlusions,
    needs_vlm,
    run_page_completeness_audit,
)
from mineru_vlm_rag.quality.completeness import normalize_text, number_tokens
from mineru_vlm_rag.settings import Settings
from mineru_vlm_rag.tables import (
    compare_table_html,
    continuation_score,
    create_table_tiles,
    merge_table_fragments,
    merge_table_html,
    table_continuation_evidence,
    table_header_text,
)
from mineru_vlm_rag.workflow import WorkflowRecorder


LOGGER = logging.getLogger(__name__)


def _canonical_text(value: str) -> str:
    """Compare presentation variants without discarding semantic characters."""
    return re.sub(r"[$\\{}^]", "", normalize_text(value))


def _counter_subset(left: Counter[str], right: Counter[str]) -> bool:
    return all(right[token] >= count for token, count in left.items())


def _mineru_run_metadata_path(work_dir: Path) -> Path:
    return work_dir / "mineru" / "run_metadata.json"


def _write_mineru_run_metadata(
    work_dir: Path,
    *,
    batch_id: str,
    archive_path: Path,
    extracted_dir: Path,
    source_pdf: Path,
    submission_pdf: Path,
    data_id: str,
) -> dict[str, Any]:
    """Persist MinerU provenance before downstream work can be interrupted."""
    metadata = {
        "batch_id": batch_id,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "extracted_dir": str(extracted_dir),
        "source_pdf": str(source_pdf),
        "source_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "submission_pdf": str(submission_pdf),
        "submission_sha256": hashlib.sha256(submission_pdf.read_bytes()).hexdigest(),
        "data_id": data_id,
    }
    path = _mineru_run_metadata_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _read_mineru_run_metadata(work_dir: Path) -> dict[str, Any]:
    path = _mineru_run_metadata_path(work_dir)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("[MinerU] invalid run metadata path=%s", path)
        return {}
    return value if isinstance(value, dict) else {}


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _task_for_block(block: Block) -> str:
    return {
        BlockType.TABLE: "table_to_cells",
        BlockType.CHART: "visual_analyze",
        BlockType.IMAGE: "visual_analyze",
        BlockType.FORMULA: "formula_to_latex",
        BlockType.SEAL: "seal_or_stamp",
    }.get(block.block_type, "ocr_verify")


def _append_tags(block: Block, key: str, *tags: str) -> None:
    existing = list(block.metadata.get(key) or [])
    block.metadata[key] = list(dict.fromkeys([*existing, *[tag for tag in tags if tag]]))


def _vlm_visual_type(result: VLMResult) -> str:
    value = result.structured_data.get("visual_type") or result.structured_data.get("type") or ""
    normalized = str(value).strip().lower()
    return {
        "表格": "table", "图表": "chart", "流程图": "diagram", "照片": "photo", "印章": "seal",
    }.get(normalized, normalized)


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是", "同一张表"}
    return False


def _exception_details(exc: Exception) -> dict[str, Any]:
    """Return useful provider diagnostics without logging request content or keys."""
    details: dict[str, Any] = {"error_type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if status_code is not None:
        details["http_status"] = status_code
    if request_id:
        details["request_id"] = str(request_id)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("code") is not None:
            details["provider_code"] = body["code"]
        if body.get("message"):
            details["provider_message"] = str(body["message"])[:300]
    return details


class IngestionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mysql = MySQLRepository(settings.mysql_dsn)
        # Parsing and VLM repair can take minutes.  Opening Milvus Lite here
        # leaves an idle gRPC channel sending 10-second keepalives throughout
        # that work, which triggers ENHANCE_YOUR_CALM/too_many_pings.  Connect
        # only when vectors are actually ready to be written.
        self.milvus: MilvusRepository | None = None
        self.mineru = MinerUClient(
            token=settings.mineru_api_token,
            api_base=settings.mineru["api_base"],
            model_version=settings.mineru["model_version"],
            language=settings.mineru["language"],
            enable_table=bool(settings.mineru["enable_table"]),
            enable_formula=bool(settings.mineru["enable_formula"]),
            is_ocr=bool(settings.mineru["is_ocr"]),
            poll_interval_seconds=float(settings.mineru["poll_interval_seconds"]),
            timeout_seconds=float(settings.mineru["timeout_seconds"]),
        )
        self.silicon = SiliconFlowClient(
            api_key=settings.siliconflow_api_key,
            base_url=settings.vlm["api_base"],
            vlm_model=settings.vlm_model,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            temperature=float(settings.vlm["temperature"]),
            max_tokens=int(settings.vlm["max_tokens"]),
            image_detail=str(settings.vlm["image_detail"]),
            max_retries=int(settings.vlm["max_retries"]),
        )
        self.detection_context = DetectionContext(
            repeated_char_ratio=float(settings.quality["repeated_char_ratio"]),
            blur_variance_threshold=float(settings.quality["blur_variance_threshold"]),
            low_contrast_std_threshold=float(settings.quality["low_contrast_std_threshold"]),
            dark_region_ratio=float(settings.quality["dark_region_ratio"]),
        )

    def initialize(self) -> None:
        self.mysql.initialize()
        LOGGER.info("[MySQL] schema ready")

    def _get_milvus(self) -> MilvusRepository:
        if self.milvus is None:
            LOGGER.info("[Milvus] opening lazy connection uri=%s", self.settings.milvus_uri)
            self.milvus = MilvusRepository(
                self.settings.milvus_uri,
                collection_prefix=str(
                    self.settings.pipeline.get("milvus_collection_prefix", "document_chunks_v3")
                ),
            )
        return self.milvus

    def _asset_path(self, document: Document, block: Block) -> Path | None:
        if not block.asset_id:
            return None
        for asset in document.assets:
            if asset.asset_id == block.asset_id and asset.source_path:
                path = Path(asset.source_path)
                if path.exists():
                    return path
        return None

    def _register_crop(self, document: Document, block: Block, path: Path) -> Asset:
        data = path.read_bytes()
        with Image.open(path) as image:
            width, height = image.size
        asset = Asset(
            document_id=document.document_id,
            page_no=block.page_no,
            kind="vlm_crop",
            mime_type="image/png",
            sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
            data=data,
            source_path=str(path),
        )
        document.assets.append(asset)
        block.metadata["vlm_crop_asset_id"] = asset.asset_id
        if not block.asset_id:
            block.asset_id = asset.asset_id
        return asset

    def _ensure_block_image(
        self,
        document: Document,
        block: Block,
        pdf_path: Path,
        work_dir: Path,
    ) -> Path:
        existing = self._asset_path(document, block)
        if existing:
            block.metadata.setdefault("vlm_image_mode", "mineru_asset")
            LOGGER.info(
                "[图片准备] page=%s block=%s mode=mineru_asset path=%s bytes=%s",
                block.page_no,
                block.block_id,
                existing,
                existing.stat().st_size,
            )
            return existing
        page = next(value for value in document.pages if value.page_no == block.page_no)
        page_image = render_page(
            pdf_path, block.page_no, work_dir / "pages", dpi=int(self.settings.pipeline["render_dpi"])
        )
        crop_path = work_dir / "crops" / f"{block.block_id}.png"
        source_width = page.metadata.get("mineru_coordinate_width") or page.width
        source_height = page.metadata.get("mineru_coordinate_height") or page.height
        crop_details: dict[str, Any] = {}
        render_block(
            page_image,
            block.bbox,
            source_width,
            source_height,
            crop_path,
            crop_details=crop_details,
        )
        block.metadata["vlm_image_mode"] = crop_details.get("mode", "unknown")
        block.metadata["vlm_crop_details"] = crop_details
        self._register_crop(document, block, crop_path)
        LOGGER.info(
            "[图片准备] page=%s block=%s mode=%s bbox=%s pixel_bbox=%s output_size=%s path=%s",
            block.page_no,
            block.block_id,
            crop_details.get("mode"),
            crop_details.get("bbox"),
            crop_details.get("pixel_bbox"),
            crop_details.get("output_size"),
            crop_path,
        )
        return crop_path

    def _apply_vlm_result(self, block: Block, result: VLMResult, *, prompt_version: str = "v2") -> None:
        text_like = block.block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.FOOTNOTE}
        mineru_primary = text_like and bool(block.raw_content.strip())
        if not result.readable:
            if mineru_primary:
                # A failed secondary reader must never destroy a non-empty
                # primary MinerU result. Keep the evidence, block automatic
                # chunking, and wait for a reliable tie-breaker.
                block.resolved_content = block.raw_content
                block.source = "mineru"
                block.status = ResolutionStatus.REVIEW
                block.metadata["chunk_eligible"] = False
                _append_tags(block, "verification_tags", "vlm_unreadable_mineru_preserved")
            else:
                block.status = ResolutionStatus.UNREADABLE
                block.source = "qwen"
            block.revisions.append(
                Revision(
                    source="qwen",
                    content="",
                    structured_data=result.model_dump(mode="json"),
                    model_name=self.settings.vlm_model,
                    prompt_version=prompt_version,
                )
            )
            return
        if block.block_type == BlockType.TABLE:
            content = result.html or result.text or json.dumps(result.structured_data, ensure_ascii=False)
        elif block.block_type == BlockType.FORMULA:
            content = result.latex or result.text
        elif block.block_type in {BlockType.IMAGE, BlockType.CHART}:
            content = result.summary or result.text or json.dumps(result.structured_data, ensure_ascii=False)
        else:
            content = result.text or result.summary
        if content.strip():
            candidate = content.strip()
            if mineru_primary:
                primary = block.raw_content.strip()
                primary_norm = _canonical_text(primary)
                candidate_norm = _canonical_text(candidate)
                comparison = block.metadata.get("pdf_text_comparison") or {}
                text_layer_reference = str(comparison.get("reference_excerpt") or "").strip()
                reference_norm = _canonical_text(text_layer_reference)
                agrees_with_mineru = bool(primary_norm) and primary_norm == candidate_norm
                candidate_reference_similarity = (
                    SequenceMatcher(None, candidate_norm, reference_norm, autojunk=False).ratio()
                    if candidate_norm and reference_norm else 0.0
                )
                primary_numbers = Counter(number_tokens(primary))
                candidate_numbers = Counter(number_tokens(candidate))
                # For an automatic addition the literal MinerU result must be
                # present byte-for-byte. Canonical similarity is sufficient
                # for an "agreed" verdict because that branch keeps MinerU,
                # but it must never authorize a rewritten replacement.
                primary_position = candidate.find(primary) if primary else -1
                added_text = (
                    candidate[:primary_position] + candidate[primary_position + len(primary):]
                    if primary_position >= 0 else ""
                )
                verified_addition = bool(
                    primary_position >= 0
                    and added_text.strip()
                    and _canonical_text(added_text) not in primary_norm
                    and candidate_reference_similarity >= 0.98
                    and _counter_subset(primary_numbers, candidate_numbers)
                )
                if agrees_with_mineru:
                    block.resolved_content = primary
                    block.source = "mineru+qwen_agreed"
                    block.status = ResolutionStatus.ACCEPTED
                    block.metadata["chunk_eligible"] = True
                    _append_tags(block, "verification_tags", "mineru_vlm_text_agreed")
                    text_decision = "preserve_mineru_agreed"
                elif verified_addition:
                    # Existing MinerU text must remain an intact substring;
                    # only independently corroborated prefix/suffix additions
                    # may be accepted automatically.
                    block.resolved_content = candidate
                    block.source = "qwen+pdf_text_layer_agreed_addition"
                    block.status = ResolutionStatus.REPAIRED
                    block.metadata["chunk_eligible"] = True
                    _append_tags(block, "processing_tags", "non_destructive_text_addition")
                    _append_tags(block, "verification_tags", "vlm_pdf_text_addition_agreed")
                    text_decision = "accept_verified_addition"
                else:
                    block.resolved_content = primary
                    block.source = "mineru"
                    block.status = ResolutionStatus.REVIEW
                    block.metadata["chunk_eligible"] = False
                    block.issue_flags = list(dict.fromkeys([*block.issue_flags, "TXT-003"]))
                    _append_tags(block, "verification_tags", "vlm_conflict_mineru_preserved")
                    text_decision = "preserve_mineru_review"
                block.metadata["vlm_text_comparison"] = {
                    "mineru_preserved": not verified_addition,
                    "agrees_with_mineru": agrees_with_mineru,
                    "verified_addition": verified_addition,
                    "candidate_reference_similarity": round(candidate_reference_similarity, 6),
                    "mineru_number_count": sum(primary_numbers.values()),
                    "candidate_number_count": sum(candidate_numbers.values()),
                }
                LOGGER.info(
                    "[文本裁决] page=%s block=%s decision=%s mineru_chars=%s vlm_chars=%s "
                    "reference_similarity=%.4f",
                    block.page_no,
                    block.block_id,
                    text_decision,
                    len(primary),
                    len(candidate),
                    candidate_reference_similarity,
                )
            else:
                block.resolved_content = candidate
                block.source = "qwen"
                block.status = ResolutionStatus.REPAIRED
                _append_tags(block, "processing_tags", "vlm_repair")
                _append_tags(block, "verification_tags", "single_vlm_source")
        else:
            if mineru_primary:
                block.resolved_content = block.raw_content
                block.source = "mineru"
                block.metadata["chunk_eligible"] = False
                _append_tags(block, "verification_tags", "vlm_empty_mineru_preserved")
            block.status = ResolutionStatus.REVIEW
        block.revisions.append(
            Revision(
                source="qwen",
                content=content,
                structured_data=result.model_dump(mode="json"),
                model_name=self.settings.vlm_model,
                prompt_version=prompt_version,
            )
        )

    def _apply_table_crosscheck(
        self,
        block: Block,
        result: VLMResult,
        *,
        mineru_html: str | None = None,
        prompt_version: str = "table-cells-v2",
    ) -> dict[str, Any]:
        candidate = (result.html or result.text or "").strip()
        block.revisions.append(
            Revision(
                source="qwen_table_crosscheck",
                content=candidate,
                structured_data=result.model_dump(mode="json"),
                model_name=self.settings.vlm_model,
                prompt_version=prompt_version,
            )
        )
        _append_tags(block, "processing_tags", "vlm_table_cell_crosscheck")
        if not result.readable or not candidate:
            block.status = ResolutionStatus.UNREADABLE if not result.readable else ResolutionStatus.REVIEW
            block.metadata["chunk_eligible"] = False
            _append_tags(block, "verification_tags", "table_crosscheck_unavailable")
            return {"comparable": False, "agreed": False, "reason": "vlm_unreadable_or_empty"}

        reference = (mineru_html if mineru_html is not None else block.raw_content).strip()
        comparison = compare_table_html(reference, candidate) if reference else {
            "comparable": False,
            "agreed": False,
            "reason": "mineru_empty",
            "conflicts": [],
        }
        block.metadata["table_crosscheck"] = comparison
        if comparison.get("agreed"):
            # Preserve the primary MinerU representation and retain Qwen as an
            # independent revision. Agreement is evidence, not ground truth.
            block.resolved_content = reference
            block.status = ResolutionStatus.REPAIRED
            block.source = "mineru+qwen_agreed"
            block.metadata["chunk_eligible"] = True
            block.issue_flags = [flag for flag in block.issue_flags if flag != "TAB-010"]
            stale_tags = {"table_crosscheck_unavailable", "dual_source_conflict", "mineru_empty"}
            block.metadata["verification_tags"] = [
                tag for tag in (block.metadata.get("verification_tags") or []) if tag not in stale_tags
            ]
            _append_tags(block, "verification_tags", "dual_source_agreed")
            if comparison.get("format_difference_count"):
                _append_tags(block, "verification_tags", "format_normalized_agreement")
        else:
            block.resolved_content = reference or candidate
            block.status = ResolutionStatus.REVIEW
            block.metadata["chunk_eligible"] = False
            block.issue_flags = list(dict.fromkeys([*block.issue_flags, "TAB-010"]))
            _append_tags(
                block,
                "verification_tags",
                "mineru_empty" if not reference else "dual_source_conflict",
            )
        return comparison

    def _analyze_table_tiles(
        self,
        block: Block,
        page: Any,
        image_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
    ) -> VLMResult:
        LOGGER.info("[表格复核] structure start page=%s block=%s", page.page_no, block.block_id)
        structure = self.silicon.analyze_images(
            [image_path],
            "table_structure",
            context=(
                f"页码={page.page_no}; MinerU候选表格={block.content[:3000]}; "
                "本次只定位标题、表头、数据区和脚注。"
            ),
        )
        block.revisions.append(
            Revision(
                source="qwen_table_structure",
                content=structure.summary or structure.text,
                structured_data=structure.model_dump(mode="json"),
                model_name=self.settings.vlm_model,
                prompt_version="table-structure-v2",
            )
        )
        _append_tags(block, "processing_tags", "table_structure_detected")
        pipeline_config = getattr(self.settings, "pipeline", {})
        tiles = create_table_tiles(
            image_path,
            work_dir / "table_tiles" / block.block_id,
            structure.structured_data,
            max_tile_height=int(pipeline_config.get("table_tile_max_height", 900)),
            overlap=int(pipeline_config.get("table_tile_overlap", 100)),
        )
        fragments: list[str] = []
        fragment_header_rows: list[int] = []
        failures: list[dict[str, Any]] = []
        for index, tile in enumerate(tiles, start=1):
            LOGGER.info(
                "[表格复核] tile start page=%s block=%s tile=%s/%s bytes=%s",
                page.page_no, block.block_id, index, len(tiles), tile.stat().st_size,
            )
            started_at = time.monotonic()
            try:
                result = self.silicon.analyze_images(
                    [tile],
                    "table_to_cells",
                    context=(
                        f"页码={page.page_no}; 数据分片={index}/{len(tiles)}; "
                        f"表结构={json.dumps(structure.structured_data, ensure_ascii=False)[:5000]}; "
                        "只抄录本分片可见单元格；重复表头保留；小数位逐位抄录。"
                    ),
                )
                if result.readable and result.html.strip():
                    fragments.append(result.html.strip())
                    fragment_header_rows.append(int(result.structured_data.get("header_rows") or 1))
                    decision = "done"
                else:
                    decision = "unreadable"
                    failures.append({"tile": index, "reason": decision})
                recorder.record(
                    "table_recovery", "vlm_table_tile", decision,
                    page_no=page.page_no, block_id=block.block_id,
                    input_refs=[str(tile)],
                    details={
                        "tile_index": index,
                        "tile_count": len(tiles),
                        "duration_seconds": round(time.monotonic() - started_at, 3),
                        "output_length": len(result.html),
                    },
                )
            except Exception as exc:
                failure = {"tile": index, **_exception_details(exc)}
                failures.append(failure)
                recorder.record(
                    "table_recovery", "vlm_table_tile", "review",
                    page_no=page.page_no, block_id=block.block_id,
                    input_refs=[str(tile)], details=failure,
                )
                LOGGER.warning(
                    "[表格复核] tile failed page=%s block=%s tile=%s/%s error=%s",
                    page.page_no, block.block_id, index, len(tiles), failure,
                )
        merged = merge_table_fragments(fragments, header_rows=fragment_header_rows)
        recorder.record(
            "table_recovery", "merge_table_tiles", "done" if merged and not failures else "review",
            page_no=page.page_no, block_id=block.block_id,
            input_refs=[str(tile) for tile in tiles],
            details={
                "tile_count": len(tiles),
                "successful_tile_count": len(fragments),
                "failure_count": len(failures),
                "failures": failures,
                "merged_length": len(merged),
            },
        )
        return VLMResult(
            task="table_to_cells",
            readable=bool(merged) and not failures,
            html=merged,
            structured_data={
                "structure": structure.structured_data,
                "tile_count": len(tiles),
                "successful_tile_count": len(fragments),
                "failures": failures,
            },
            uncertainty=[] if not failures else ["one_or_more_table_tiles_failed"],
        )

    def _secondary_mineru_table(
        self,
        document: Document,
        block: Block,
        image_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
    ) -> str:
        """Submit a standalone table image as a one-page PDF to MinerU."""
        check_dir = work_dir / "table_recheck" / block.block_id
        check_dir.mkdir(parents=True, exist_ok=True)
        mini_pdf = check_dir / "expanded-table.pdf"
        with Image.open(image_path) as source:
            source.convert("RGB").save(mini_pdf, format="PDF", resolution=300.0)
        started_at = time.monotonic()
        result = self.mineru.parse_pdf(
            mini_pdf,
            check_dir / "mineru",
            data_id=f"table-recheck-{document.document_id[:12]}-{block.block_id}",
        )
        candidate_document = Document(
            document_id=f"{document.document_id}-table-{block.block_id}",
            file_name=mini_pdf.name,
            source_path=str(mini_pdf),
            sha256=hashlib.sha256(mini_pdf.read_bytes()).hexdigest(),
            page_count=1,
        )
        parse_mineru_output(candidate_document, result.extracted_dir)
        tables = [
            candidate
            for page in candidate_document.pages
            for candidate in page.blocks
            if candidate.block_type == BlockType.TABLE and candidate.content.strip()
        ]
        content = max(tables, key=lambda candidate: len(candidate.content)).content if tables else ""
        recorder.record(
            "table_recovery",
            "secondary_mineru_table",
            "done" if content else "empty",
            page_no=block.page_no,
            block_id=block.block_id,
            issue_flags=["TAB-009"],
            input_refs=[str(mini_pdf)],
            output_refs=[str(result.archive_path)],
            details={
                "batch_id": result.batch_id,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "table_count": len(tables),
                "output_length": len(content),
            },
        )
        block.revisions.append(
            Revision(
                source="mineru_secondary_table",
                content=content,
                structured_data={"batch_id": result.batch_id, "archive_path": str(result.archive_path)},
                prompt_version="mineru-table-recheck-v1",
            )
        )
        return content

    def _recover_misclassified_table(
        self,
        document: Document,
        page: Any,
        block: Block,
        image_path: Path,
        structure_result: VLMResult,
        work_dir: Path,
        recorder: WorkflowRecorder,
    ) -> None:
        original_type = block.block_type.value
        block.issue_flags = list(dict.fromkeys([*block.issue_flags, "TAB-009"]))
        block.metadata["mineru_original_type"] = original_type
        block.metadata["table_structure"] = structure_result.structured_data
        _append_tags(block, "processing_tags", "visual_type_reclassified", "table_structure_detected")
        block.revisions.append(
            Revision(
                source="qwen_table_structure",
                content=structure_result.summary or structure_result.text,
                structured_data=structure_result.model_dump(mode="json"),
                model_name=self.settings.vlm_model,
                prompt_version="visual-structure-v2",
            )
        )
        tiles = create_table_tiles(
            image_path,
            work_dir / "table_tiles" / block.block_id,
            structure_result.structured_data,
        )
        fragments: list[str] = []
        fragment_header_rows: list[int] = []
        for index, tile in enumerate(tiles, start=1):
            LOGGER.info(
                "[表格恢复] VLM tile start page=%s block=%s tile=%s/%s path=%s",
                page.page_no, block.block_id, index, len(tiles), tile,
            )
            result = self.silicon.analyze_images(
                [tile],
                "table_to_cells",
                context=(
                    f"页码={page.page_no}; 数据分片={index}/{len(tiles)}; "
                    f"表结构={json.dumps(structure_result.structured_data, ensure_ascii=False)[:5000]}; "
                    "只抄录当前分片可见单元格，重复表头必须保留。"
                ),
            )
            if result.readable and result.html.strip():
                fragments.append(result.html.strip())
                fragment_header_rows.append(int(result.structured_data.get("header_rows") or 1))
        merged = merge_table_fragments(fragments, header_rows=fragment_header_rows)
        qwen_result = VLMResult(
            task="table_to_cells",
            readable=bool(merged),
            html=merged,
            structured_data={
                "visual_type": "table",
                "tile_count": len(tiles),
                "successful_tile_count": len(fragments),
            },
            uncertainty=[] if len(fragments) == len(tiles) else ["one_or_more_table_tiles_failed"],
        )
        mineru_html = ""
        try:
            mineru_html = self._secondary_mineru_table(document, block, image_path, work_dir, recorder)
        except Exception as exc:
            recorder.record(
                "table_recovery", "secondary_mineru_table", "review",
                page_no=page.page_no, block_id=block.block_id, issue_flags=["TAB-009"],
                details=_exception_details(exc),
            )
        block.block_type = BlockType.TABLE
        comparison = self._apply_table_crosscheck(
            block,
            qwen_result,
            mineru_html=mineru_html,
            prompt_version="misclassified-table-tiles-v2",
        )
        recorder.record(
            "table_recovery",
            "compare_mineru_qwen_cells",
            "agreed" if comparison.get("agreed") else "review",
            page_no=page.page_no,
            block_id=block.block_id,
            issue_flags=block.issue_flags,
            input_refs=[str(path) for path in tiles],
            details={
                "original_type": original_type,
                "tile_count": len(tiles),
                "successful_tile_count": len(fragments),
                **comparison,
            },
        )

    def _process_blocks(
        self,
        document: Document,
        pdf_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
        use_vlm: bool,
    ) -> None:
        work_unit = max(1, int(self.settings.pipeline["page_work_unit"]))
        pages = sorted(document.pages, key=lambda value: value.page_no)
        for offset in range(0, len(pages), work_unit):
            unit_pages = pages[offset:offset + work_unit]
            unit_index = offset // work_unit + 1
            LOGGER.info(
                "[页面工作单元] start unit=%s pages=%s-%s total_pages=%s",
                unit_index,
                unit_pages[0].page_no,
                unit_pages[-1].page_no,
                len(pages),
            )
            recorder.record(
                "page_work_unit", "process_pages", "started",
                details={
                    "unit_index": unit_index,
                    "page_start": unit_pages[0].page_no,
                    "page_end": unit_pages[-1].page_no,
                    "configured_size": work_unit,
                },
            )
            for page in unit_pages:
                LOGGER.info(
                    "[页面处理] page=%s/%s blocks=%s",
                    page.page_no,
                    len(pages),
                    len(page.blocks),
                )
                rebuild_reading_order(page)
                for block in page.blocks:
                    self._process_block(document, page, block, pdf_path, work_dir, recorder, use_vlm)
            recorder.record(
                "page_work_unit", "process_pages", "completed",
                details={
                    "unit_index": unit_index,
                    "page_start": unit_pages[0].page_no,
                    "page_end": unit_pages[-1].page_no,
                    "boundary_table_check_deferred": True,
                },
            )
            LOGGER.info(
                "[页面工作单元] done unit=%s pages=%s-%s",
                unit_index,
                unit_pages[0].page_no,
                unit_pages[-1].page_no,
            )

    def _add_page_occlusion_blocks(
        self,
        document: Document,
        pdf_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
    ) -> None:
        for page in document.pages:
            page_image = render_page(
                pdf_path,
                page.page_no,
                work_dir / "pages",
                dpi=int(self.settings.pipeline["render_dpi"]),
            )
            regions = detect_solid_occlusions(page_image)
            if not regions:
                continue
            with Image.open(page_image) as image:
                for index, region in enumerate(regions, start=1):
                    if not page.width or not page.height:
                        continue
                    bbox = BoundingBox(
                        x0=region.x / image.width * page.width,
                        y0=region.y / image.height * page.height,
                        x1=(region.x + region.width) / image.width * page.width,
                        y1=(region.y + region.height) / image.height * page.height,
                    )
                    block = Block(
                        document_id=document.document_id,
                        page_no=page.page_no,
                        block_type=BlockType.TEXT,
                        bbox=bbox,
                        reading_order=len(page.blocks),
                        raw_content="",
                        resolved_content="[该区域被遮挡或不可读，无法确认内容]",
                        source="quality_gate",
                        issue_flags=["SRC-004"],
                        status=ResolutionStatus.UNREADABLE,
                        metadata={
                            "detector": "solid_dark_rectangle",
                            "fill_ratio": round(region.fill_ratio, 4),
                            "unreadable_reason": "solid_occlusion",
                        },
                        revisions=[Revision(
                            source="quality_gate",
                            content="[该区域被遮挡或不可读，无法确认内容]",
                            structured_data={
                                "detector": "solid_dark_rectangle",
                                "fill_ratio": round(region.fill_ratio, 4),
                            },
                        )],
                    )
                    crop_path = work_dir / "page_quality" / f"page-{page.page_no:04d}-occlusion-{index}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((region.x, region.y, region.x + region.width, region.y + region.height)).save(crop_path)
                    asset = self._register_crop(document, block, crop_path)
                    asset.kind = "occlusion_evidence"
                    page.blocks.append(block)
                    recorder.record(
                        "page_quality", "detect_solid_occlusion", "unreadable",
                        page_no=page.page_no,
                        block_id=block.block_id,
                        issue_flags=["SRC-004"],
                        input_refs=[str(page_image)],
                        output_refs=[asset.asset_id],
                        details={
                            "pixel_bbox": [region.x, region.y, region.width, region.height],
                            "fill_ratio": round(region.fill_ratio, 4),
                        },
                    )
                    LOGGER.warning(
                        "[遮挡检测] page=%s block=%s unreadable fill_ratio=%s crop=%s",
                        page.page_no,
                        block.block_id,
                        round(region.fill_ratio, 4),
                        crop_path,
                    )

    def _process_block(
        self,
        document: Document,
        page: Any,
        block: Block,
        pdf_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
        use_vlm: bool,
    ) -> None:
        asset_path = self._asset_path(document, block)
        detected_flags = detect_block_issues(block, self.detection_context, asset_path)
        flags = list(dict.fromkeys([*block.issue_flags, *detected_flags]))
        if block.bbox and block.bbox.height > max(block.bbox.width * 2.5, 80):
            flags.append("LAY-002")
            block.metadata.setdefault("original_orientation", "vertical_or_rotated")
        block.issue_flags = flags
        pdf_text_verified = (
            block.source == "pdf_text_layer"
            and (
                block.metadata.get("text_layer_reliable")
                or "pdf_text_layer_recovered" in (block.metadata.get("verification_tags") or [])
            )
        )
        gate_decision = (
            "unreadable" if "SRC-004" in flags
            else "accept_pdf_text" if pdf_text_verified
            else "vlm" if needs_vlm(flags)
            else "accept"
        )
        recorder.record(
            "quality_gate", "detect_block_issues", gate_decision,
            page_no=page.page_no, block_id=block.block_id, issue_flags=flags,
            input_refs=[block.asset_id] if block.asset_id else [],
            details={"block_type": block.block_type.value, "content_length": len(block.content)},
        )
        if gate_decision != "accept":
            LOGGER.info(
                "[质量门控] page=%s block=%s type=%s decision=%s flags=%s content_chars=%s",
                page.page_no,
                block.block_id,
                block.block_type.value,
                gate_decision,
                ",".join(flags) or "-",
                len(block.content),
            )
        if "SRC-004" in flags:
            block.status = ResolutionStatus.UNREADABLE
            block.resolved_content = "[该区域被遮挡或不可读，无法确认内容]"
            recorder.record(
                "resolution", "mark_unreadable", "unreadable",
                page_no=page.page_no, block_id=block.block_id, issue_flags=flags,
                details={"reason": "dark_or_occluded_region"},
            )
            return
        if pdf_text_verified:
            recorder.record(
                "resolution", "accept_pdf_text_recovery", "repaired",
                page_no=page.page_no, block_id=block.block_id, issue_flags=flags,
                details={"reason": "reliable_text_layer_plus_visible_glyph_evidence"},
            )
            return
        if not use_vlm or not needs_vlm(flags):
            return
        task = _task_for_block(block)
        image_path = self._ensure_block_image(document, block, pdf_path, work_dir)
        image_paths = [image_path]
        if any(flag in {"SRC-002", "SRC-003"} for flag in flags):
            enhanced_path = work_dir / "enhanced" / f"{block.block_id}.png"
            enhance_image(image_path, enhanced_path)
            image_paths.append(enhanced_path)
        caption = block.metadata.get("caption", "")
        context = (
            f"页码={page.page_no}; block_type={block.block_type.value}; "
            f"邻近标题={caption}; MinerU候选={block.content[:2000]}"
        )
        if len(image_paths) == 2:
            context += "; 第1张为原始裁剪，第2张为增强裁剪，冲突内容必须写入uncertainty"
        recorder.record(
            "image_preparation",
            "prepare_vlm_images",
            "ready",
            page_no=page.page_no,
            block_id=block.block_id,
            issue_flags=flags,
            input_refs=[str(path) for path in image_paths],
            output_refs=[block.metadata.get("vlm_crop_asset_id") or block.asset_id or ""],
            details={
                "task": task,
                "block_type": block.block_type.value,
                "image_mode": block.metadata.get("vlm_image_mode"),
                "image_count": len(image_paths),
                "image_bytes": [path.stat().st_size for path in image_paths],
                "enhanced": len(image_paths) == 2,
                "crop_details": block.metadata.get("vlm_crop_details", {}),
            },
        )
        LOGGER.info(
            "[VLM] start page=%s block=%s task=%s mode=%s images=%s",
            page.page_no,
            block.block_id,
            task,
            block.metadata.get("vlm_image_mode"),
            len(image_paths),
        )
        started_at = time.monotonic()
        try:
            if task == "table_to_cells":
                result = self._analyze_table_tiles(block, page, image_path, work_dir, recorder)
            else:
                result = self.silicon.analyze_images(image_paths, task, context=context)
            visual_type = _vlm_visual_type(result) if task == "visual_analyze" else ""
            if task == "visual_analyze" and (visual_type == "table" or "<table" in result.html.lower()):
                self._recover_misclassified_table(
                    document, page, block, image_path, result, work_dir, recorder,
                )
            elif task == "table_to_cells":
                self._apply_table_crosscheck(block, result)
            else:
                self._apply_vlm_result(block, result)
            duration_seconds = round(time.monotonic() - started_at, 3)
            recorder.record(
                "vlm", task, block.status.value,
                page_no=page.page_no, block_id=block.block_id, issue_flags=flags,
                input_refs=[str(path) for path in image_paths],
                output_refs=[block.metadata.get("vlm_crop_asset_id") or block.asset_id or ""],
                details={
                    "readable": result.readable,
                    "output_length": len(block.content),
                    "image_mode": block.metadata.get("vlm_image_mode"),
                    "crop_details": block.metadata.get("vlm_crop_details", {}),
                    "duration_seconds": duration_seconds,
                    "uncertainty_count": len(result.uncertainty),
                    "visual_type": visual_type or None,
                },
            )
            LOGGER.info(
                "[VLM] done page=%s block=%s task=%s status=%s duration=%.3fs output_chars=%s uncertainty=%s",
                page.page_no,
                block.block_id,
                task,
                block.status.value,
                duration_seconds,
                len(block.content),
                len(result.uncertainty),
            )
        except Exception as exc:
            block.status = ResolutionStatus.REVIEW
            duration_seconds = round(time.monotonic() - started_at, 3)
            error_details = {
                **_exception_details(exc),
                "duration_seconds": duration_seconds,
                "image_mode": block.metadata.get("vlm_image_mode"),
                "crop_details": block.metadata.get("vlm_crop_details", {}),
            }
            recorder.record(
                "vlm", task, "review",
                page_no=page.page_no, block_id=block.block_id, issue_flags=flags,
                input_refs=[str(path) for path in image_paths], details=error_details,
            )
            LOGGER.warning(
                "[VLM] review page=%s block=%s task=%s duration=%.3fs error=%s status=%s provider_code=%s message=%s",
                page.page_no,
                block.block_id,
                task,
                duration_seconds,
                error_details["error_type"],
                error_details.get("http_status"),
                error_details.get("provider_code"),
                error_details.get("provider_message"),
            )

    def _table_similarity(self, left: Block, right: Block) -> float | None:
        headers = [table_header_text(left.content), table_header_text(right.content)]
        if not all(headers):
            return None
        vectors = self.silicon.embed(headers, dimensions=int(self.settings.embedding["dimensions"]))
        return _cosine(vectors[0], vectors[1]) if len(vectors) == 2 else None

    def _merge_cross_page_tables(
        self,
        document: Document,
        pdf_path: Path,
        work_dir: Path,
        recorder: WorkflowRecorder,
        use_vlm: bool,
    ) -> None:
        auto_threshold = float(self.settings.quality["table_continuation_auto_threshold"])
        vlm_threshold = float(self.settings.quality["table_continuation_vlm_threshold"])
        pages = sorted(document.pages, key=lambda page: page.page_no)
        blocks_by_id = {
            block.block_id: block
            for page in pages
            for block in page.blocks
        }
        for previous_page, current_page in zip(pages, pages[1:]):
            previous_tables = [b for b in previous_page.blocks if b.block_type == BlockType.TABLE]
            current_tables = [b for b in current_page.blocks if b.block_type == BlockType.TABLE and not b.metadata.get("skip_chunk")]
            if not previous_tables or not current_tables:
                continue
            previous_tail = previous_tables[-1]
            merged_into = previous_tail.metadata.get("merged_into")
            previous = blocks_by_id.get(merged_into, previous_tail)
            current = current_tables[0]
            evidence = table_continuation_evidence(
                previous,
                current,
                previous_page,
                current_page,
            )
            semantic = None
            if evidence["edge_continuity"] and not evidence["hard_negative"]:
                try:
                    semantic = self._table_similarity(previous, current)
                except Exception as exc:
                    LOGGER.warning(
                        "[跨页表] header embedding unavailable pages=%s-%s error=%s",
                        previous.page_no,
                        current.page_no,
                        type(exc).__name__,
                    )
            score = continuation_score(previous, current, semantic)
            should_merge = bool(evidence["auto_eligible"] and score >= auto_threshold)
            decision_source = "identity_edge_rules" if evidence["hard_negative"] else "rules_embedding"
            decision_error: dict[str, Any] = {}
            if (
                not should_merge
                and use_vlm
                and evidence["edge_continuity"]
                and not evidence["hard_negative"]
                and score >= vlm_threshold
            ):
                try:
                    images = [
                        self._ensure_block_image(document, previous, pdf_path, work_dir),
                        self._ensure_block_image(document, current, pdf_path, work_dir),
                    ]
                    LOGGER.info(
                        "[跨页表] VLM decision start pages=%s-%s score=%.4f",
                        previous.page_no,
                        current.page_no,
                        score,
                    )
                    result = self.silicon.analyze_images(images, "table_continuation", context=f"规则得分={score}")
                    raw_decision = (
                        result.is_continuation
                        if result.is_continuation is not None
                        else result.structured_data.get("is_continuation")
                    )
                    should_merge = _strict_bool(raw_decision)
                    decision_source = "qwen"
                except Exception as exc:
                    decision_source = "review"
                    decision_error = _exception_details(exc)
            recorder.record(
                "cross_page_table", "decide_continuation", "merge" if should_merge else "separate",
                page_no=current.page_no, block_id=current.block_id, issue_flags=["TAB-003"],
                input_refs=[previous.block_id, current.block_id],
                details={
                    "score": score,
                    "semantic_similarity": semantic,
                    "source": decision_source,
                    "continuation_evidence": evidence,
                    **decision_error,
                },
            )
            LOGGER.info(
                "[跨页表] pages=%s-%s decision=%s score=%.4f semantic=%s source=%s",
                previous.page_no,
                current.page_no,
                "merge" if should_merge else "separate",
                score,
                semantic,
                decision_source,
            )
            if not should_merge:
                continue
            try:
                merged = merge_table_html(
                    previous.content,
                    current.content,
                    previous_page_no=previous.page_no,
                    current_page_no=current.page_no,
                )
            except Exception:
                previous.status = ResolutionStatus.REVIEW
                current.status = ResolutionStatus.REVIEW
                continue
            previous.resolved_content = merged
            previous.metadata.setdefault("pre_merge_status", previous.status.value)
            previous.status = ResolutionStatus.REPAIRED
            previous.issue_flags = list(dict.fromkeys([*previous.issue_flags, "TAB-003"]))
            previous.metadata.setdefault("source_pages", [previous.page_no])
            previous.metadata["source_pages"].append(current.page_no)
            previous.revisions.append(Revision(source="table_merge", content=merged, structured_data={"score": score}))
            current.metadata["skip_chunk"] = True
            current.metadata["merged_into"] = previous.block_id

    def ingest(
        self,
        pdf_path: Path,
        use_vlm: bool = True,
        *,
        fresh_mineru: bool = False,
        reuse_mineru: bool = False,
    ) -> dict[str, Any]:
        pdf_path = pdf_path.resolve()
        if fresh_mineru and reuse_mineru:
            raise ValueError("fresh_mineru and reuse_mineru are mutually exclusive")
        preflight = inspect_pdf(pdf_path)
        document_id = preflight.sha256
        work_dir = self.settings.project_root / "outputs" / document_id
        recorder = WorkflowRecorder(document_id, self.settings.project_root / "outputs" / "workflow_logs")
        runtime_log_path = work_dir / "runtime_logs" / f"{recorder.run_id}.log"
        runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_handler = logging.FileHandler(runtime_log_path, encoding="utf-8")
        runtime_handler.setLevel(logging.INFO)
        runtime_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        package_logger = logging.getLogger("mineru_vlm_rag")
        package_logger.addHandler(runtime_handler)
        document = Document(
            document_id=document_id,
            file_name=pdf_path.name,
            source_path=str(pdf_path),
            sha256=preflight.sha256,
            page_count=preflight.page_count,
            status="processing",
            metadata={
                "file_size": preflight.file_size,
                "encrypted": preflight.encrypted,
                "fresh_mineru": fresh_mineru,
                "reuse_mineru": reuse_mineru,
            },
        )
        recorder.record(
            "preflight", "inspect_pdf", "accepted",
            input_refs=[str(pdf_path)], details={"page_count": preflight.page_count, "file_size": preflight.file_size},
        )
        LOGGER.info(
            "[入库] start run_id=%s document=%s file=%s pages=%s bytes=%s vlm=%s runtime_log=%s",
            recorder.run_id,
            document_id,
            pdf_path,
            preflight.page_count,
            preflight.file_size,
            use_vlm,
            runtime_log_path,
        )
        try:
            if reuse_mineru:
                archive_path = work_dir / "mineru" / "mineru-result.zip"
                extracted_dir = work_dir / "mineru" / "extracted"
                if not archive_path.is_file() or not extracted_dir.is_dir():
                    raise FileNotFoundError(
                        f"Reusable MinerU result is incomplete: {work_dir / 'mineru'}"
                    )
                if not any(extracted_dir.rglob("*_content_list.json")):
                    raise FileNotFoundError(f"MinerU content_list is missing: {extracted_dir}")
                run_metadata = _read_mineru_run_metadata(work_dir)
                result = SimpleNamespace(
                    batch_id=str(run_metadata.get("batch_id") or "reused-local-result-unknown-batch"),
                    archive_path=archive_path,
                    extracted_dir=extracted_dir,
                )
                document.metadata["mineru_run_metadata"] = run_metadata
                LOGGER.info(
                    "[MinerU] reuse local result batch_id=%s archive=%s extracted=%s metadata=%s",
                    result.batch_id, archive_path, extracted_dir, _mineru_run_metadata_path(work_dir),
                )
            else:
                submission_pdf = pdf_path
                data_id = document_id
                if fresh_mineru:
                    submission_pdf, nonce = create_fresh_submission_copy(
                        pdf_path,
                        work_dir / "mineru_submission",
                    )
                    data_id = f"fresh-{nonce}"
                    document.metadata.update(
                        {
                            "mineru_submission_path": str(submission_pdf),
                            "mineru_submission_sha256": hashlib.sha256(submission_pdf.read_bytes()).hexdigest(),
                            "mineru_fresh_nonce": nonce,
                        }
                    )
                    LOGGER.info(
                        "[MinerU] fresh submission created original=%s submission=%s data_id=%s",
                        pdf_path, submission_pdf, data_id,
                    )
                LOGGER.info("[MinerU] upload/parse start pages=%s file=%s", preflight.page_count, submission_pdf)
                result = self.mineru.parse_pdf(submission_pdf, work_dir / "mineru", data_id=data_id)
                run_metadata = _write_mineru_run_metadata(
                    work_dir,
                    batch_id=result.batch_id,
                    archive_path=result.archive_path,
                    extracted_dir=result.extracted_dir,
                    source_pdf=pdf_path,
                    submission_pdf=submission_pdf,
                    data_id=data_id,
                )
                document.metadata["mineru_run_metadata"] = run_metadata
            document.mineru_batch_id = result.batch_id
            document.mineru_archive_path = str(result.archive_path)
            recorder.record(
                "mineru", "reuse_local_result" if reuse_mineru else "signed_upload_parse", "done",
                input_refs=[str(pdf_path)], output_refs=[str(result.archive_path)],
                details={"batch_id": result.batch_id},
            )
            LOGGER.info(
                "[MinerU] done batch_id=%s archive=%s extracted=%s",
                result.batch_id,
                result.archive_path,
                result.extracted_dir,
            )
            parse_mineru_output(document, result.extracted_dir)
            for page in document.pages:
                if 1 <= page.page_no <= len(preflight.page_sizes):
                    page.width, page.height = preflight.page_sizes[page.page_no - 1]
                    page.rotation = preflight.rotations[page.page_no - 1]
            block_type_counts = Counter(
                block.block_type.value for page in document.pages for block in page.blocks
            )
            recorder.record(
                "normalization",
                "parse_mineru_content_list",
                "done",
                details={
                    "page_count": len(document.pages),
                    "block_count": sum(block_type_counts.values()),
                    "asset_count": len(document.assets),
                    "block_type_counts": dict(sorted(block_type_counts.items())),
                },
            )
            LOGGER.info(
                "[结构化] pages=%s blocks=%s assets=%s block_types=%s",
                len(document.pages),
                sum(block_type_counts.values()),
                len(document.assets),
                dict(sorted(block_type_counts.items())),
            )
            LOGGER.info("[页面质量] solid-occlusion scan start pages=%s", len(document.pages))
            self._add_page_occlusion_blocks(document, pdf_path, work_dir, recorder)
            LOGGER.info("[页面完整性] reverse audit start pages=%s", len(document.pages))
            completeness_summary = run_page_completeness_audit(
                document,
                pdf_path,
                work_dir,
                render_dpi=int(self.settings.pipeline["render_dpi"]),
            )
            document.metadata["page_completeness"] = completeness_summary
            recorder.record(
                "page_completeness",
                "reverse_compare_pdf_and_mineru",
                "done",
                output_refs=[completeness_summary["findings_path"]],
                details=completeness_summary,
            )
            LOGGER.info(
                "[页面完整性] done reliable=%s findings=%s recovered=%s review=%s report=%s",
                completeness_summary["text_layer"]["reliable"],
                completeness_summary["finding_count"],
                completeness_summary["recovered_block_count"],
                completeness_summary["review_candidate_count"],
                completeness_summary["findings_path"],
            )
            mark_repeated_templates(document)
            associate_visual_context(document)
            self._process_blocks(document, pdf_path, work_dir, recorder, use_vlm)
            status_counts = Counter(
                block.status.value for page in document.pages for block in page.blocks
            )
            LOGGER.info("[复杂块处理] done status_counts=%s", dict(sorted(status_counts.items())))
            structure_summary = normalize_document_structure(document)
            recorder.record(
                "normalization", "resolve_structure", "done", details=structure_summary,
            )
            LOGGER.info("[结构顺序] normalized summary=%s", structure_summary)
            LOGGER.info("[跨页表] global adjacent-page scan start")
            self._merge_cross_page_tables(document, pdf_path, work_dir, recorder, use_vlm)
            chunks = build_chunks(
                document,
                text_size=int(self.settings.pipeline["text_chunk_chars"]),
                overlap=int(self.settings.pipeline["text_chunk_overlap"]),
                table_row_group_size=int(self.settings.pipeline.get("table_row_group_size", 6)),
            )
            document.status = "parsed"
            LOGGER.info(
                "[切块] done chunks=%s assets=%s; saving MySQL evidence graph",
                len(chunks),
                len(document.assets),
            )
            self.mysql.save_document_graph(document, chunks)
            recorder.record(
                "chunking", "build_multimodal_chunks", "done",
                details={"chunk_count": len(chunks), "asset_count": len(document.assets)},
            )
            vectors: list[list[float]] = []
            batch_size = int(self.settings.embedding["batch_size"])
            total_batches = (len(chunks) + batch_size - 1) // batch_size if chunks else 0
            for offset in range(0, len(chunks), batch_size):
                texts = [chunk.embedding_text for chunk in chunks[offset:offset + batch_size]]
                batch_index = offset // batch_size + 1
                LOGGER.info(
                    "[Embedding] start batch=%s/%s chunks=%s-%s count=%s",
                    batch_index,
                    total_batches,
                    offset + 1,
                    offset + len(texts),
                    len(texts),
                )
                embedding_started_at = time.monotonic()
                batch_vectors = self.silicon.embed(
                    texts,
                    dimensions=int(self.settings.embedding["dimensions"]),
                )
                duration_seconds = round(time.monotonic() - embedding_started_at, 3)
                vectors.extend(batch_vectors)
                vector_dimension = len(batch_vectors[0]) if batch_vectors else 0
                recorder.record(
                    "embedding",
                    "embed_batch",
                    "done",
                    details={
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "chunk_offset": offset,
                        "chunk_count": len(texts),
                        "vector_dimension": vector_dimension,
                        "duration_seconds": duration_seconds,
                    },
                )
                LOGGER.info(
                    "[Embedding] done batch=%s/%s duration=%.3fs dimension=%s",
                    batch_index,
                    total_batches,
                    duration_seconds,
                    vector_dimension,
                )
            collection = ""
            if chunks:
                LOGGER.info("[Milvus] upsert start chunks=%s vectors=%s", len(chunks), len(vectors))
                collection = self._get_milvus().upsert(chunks, vectors)
                self.mysql.mark_chunks_embedded(chunk.chunk_id for chunk in chunks)
                LOGGER.info("[Milvus] upsert done collection=%s", collection)
            document.status = "completed"
            self.mysql.update_document_status(document.document_id, document.status)
            recorder.record(
                "indexing", "embed_and_upsert", "done",
                output_refs=[collection], details={"chunk_count": len(chunks), "dimension": len(vectors[0]) if vectors else 0},
            )
            self.mysql.save_workflow_events(recorder.events)
            summary_path = recorder.write_markdown_summary()
            LOGGER.info(
                "[入库] completed run_id=%s pages=%s blocks=%s assets=%s chunks=%s collection=%s",
                recorder.run_id,
                len(document.pages),
                sum(len(page.blocks) for page in document.pages),
                len(document.assets),
                len(chunks),
                collection,
            )
            return {
                "run_id": recorder.run_id,
                "document_id": document_id,
                "status": document.status,
                "pages": len(document.pages),
                "blocks": sum(len(page.blocks) for page in document.pages),
                "assets": len(document.assets),
                "chunks": len(chunks),
                "collection": collection,
                "workflow_log": str(recorder.jsonl_path),
                "workflow_summary": str(summary_path),
                "runtime_log": str(runtime_log_path),
                "page_audit": completeness_summary,
            }
        except Exception as exc:
            document.status = "failed"
            error_details = _exception_details(exc)
            document.metadata.update(error_details)
            recorder.record(
                "pipeline",
                "ingest",
                "failed",
                output_refs=[str(runtime_log_path)],
                details=error_details,
            )
            LOGGER.exception(
                "[入库] failed run_id=%s document=%s error=%s",
                recorder.run_id,
                document_id,
                error_details,
            )
            self.mysql.save_document_graph(document, [])
            self.mysql.save_workflow_events(recorder.events)
            recorder.write_markdown_summary()
            raise
        finally:
            package_logger.removeHandler(runtime_handler)
            runtime_handler.close()

    def close(self) -> None:
        self.mineru.close()
        self.silicon.close()
        if self.milvus is not None:
            self.milvus.close()
