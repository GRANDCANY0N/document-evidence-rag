from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv import dotenv_values
from pypdf import PdfReader

from mineru_vlm_rag.adapters.mineru_client import MinerUClient
from mineru_vlm_rag.pdf import create_fresh_submission_copy


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backward-compatible private alias retained for existing callers/tests.
_create_fresh_submission_copy = create_fresh_submission_copy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_url(value: str) -> str:
    """Keep a useful URL path without persisting signed query credentials."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if not parts.scheme or not parts.netloc:
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _redact_result(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(item_key): _redact_result(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_result(item, key) for item in value]
    if isinstance(value, str) and "url" in key.lower():
        return _redact_url(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pdf_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path), strict=False).pages)


def _layout_metadata(extracted_dir: Path) -> dict[str, Any]:
    candidates = [extracted_dir / "layout.json"]
    candidates.extend(sorted(extracted_dir.rglob("*_middle.json")))
    layout_path = next((path for path in candidates if path.is_file()), None)
    if layout_path is None:
        return {"layout_json": None}
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"layout_json": str(layout_path), "read_error": f"{type(exc).__name__}: {exc}"}
    return {
        "layout_json": str(layout_path),
        "backend": layout.get("_backend"),
        "effort": layout.get("_effort"),
        "ocr_enable": layout.get("_ocr_enable"),
        "version_name": layout.get("_version_name"),
        "page_count": len(layout.get("pdf_info") or []),
    }


def run_mineru_vlm_only(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_ranges: str | None = None,
    fresh: bool = False,
    reject_same_as: Path | None = None,
    timeout_seconds: float | None = None,
    env_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run only MinerU's remote VLM parser and preserve its returned archive.

    This deliberately does not construct the ingestion pipeline, connect to
    MySQL/Milvus, call the local SiliconFlow VLM, chunk text, or embed content.
    """
    pdf_path = pdf_path.resolve()
    output_dir = output_dir.resolve()
    env_path = (env_path or PROJECT_ROOT / ".env").resolve()
    config_path = (config_path or PROJECT_ROOT / "config" / "default.yaml").resolve()

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be an existing PDF: {pdf_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new run directory so MinerU results are not overwritten."
        )

    output_dir.mkdir(parents=True)
    log_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    package_logger = logging.getLogger("mineru_vlm_rag")
    package_logger.addHandler(file_handler)

    client: MinerUClient | None = None
    verification_summary: dict[str, Any] | None = None
    try:
        env = dotenv_values(env_path)
        token = str(env.get("MINERU_API_TOKEN") or "").strip()
        if not token:
            raise ValueError(f"MINERU_API_TOKEN is missing from {env_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        mineru = config.get("mineru") or {}
        effective_timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else mineru.get("timeout_seconds", 1800)
        )
        if effective_timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        document_sha256 = _sha256(pdf_path)
        source_page_count = _pdf_page_count(pdf_path)
        submission_pdf = pdf_path
        fresh_nonce: str | None = None
        if fresh:
            submission_pdf, fresh_nonce = create_fresh_submission_copy(pdf_path, output_dir)
        submission_sha256 = _sha256(submission_pdf)
        data_id = document_sha256 if not fresh_nonce else f"fresh-{fresh_nonce}"
        baseline_path = reject_same_as.resolve() if reject_same_as is not None else None
        if baseline_path is not None and not baseline_path.is_file():
            raise ValueError(f"Baseline archive does not exist: {baseline_path}")
        baseline_sha256 = _sha256(baseline_path) if baseline_path is not None else None
        request_summary = {
            "mode": "mineru_vlm_only",
            "input_pdf": str(pdf_path),
            "input_bytes": pdf_path.stat().st_size,
            "input_page_count": source_page_count,
            "sha256": document_sha256,
            "submission_pdf": str(submission_pdf),
            "submission_sha256": submission_sha256,
            "fresh": fresh,
            "fresh_nonce": fresh_nonce,
            "data_id": data_id,
            "reject_same_as": str(baseline_path) if baseline_path else None,
            "reject_same_as_sha256": baseline_sha256,
            "output_dir": str(output_dir),
            "api_base": str(mineru.get("api_base") or "https://mineru.net"),
            "model_version": "vlm",
            "language": str(mineru.get("language") or "ch"),
            "enable_table": bool(mineru.get("enable_table", True)),
            "enable_formula": bool(mineru.get("enable_formula", True)),
            "is_ocr": bool(mineru.get("is_ocr", True)),
            "page_ranges": page_ranges,
            "timeout_seconds": effective_timeout_seconds,
            "downstream_disabled": [
                "local_qwen_vlm",
                "quality_gate",
                "cross_page_merge",
                "chunking",
                "mysql",
                "embedding",
                "milvus",
            ],
        }
        _write_json(output_dir / "request.json", request_summary)
        LOGGER.info(
            "[MinerU-only] start model=vlm pdf=%s bytes=%s sha256=%s output=%s",
            pdf_path,
            pdf_path.stat().st_size,
            document_sha256,
            output_dir,
        )
        if fresh:
            LOGGER.info(
                "[MinerU-only] fresh submission=%s sha256=%s data_id=%s baseline_sha256=%s",
                submission_pdf,
                submission_sha256,
                data_id,
                baseline_sha256 or "none",
            )
        LOGGER.info(
            "[MinerU-only] parameters language=%s ocr=%s table=%s formula=%s page_ranges=%s",
            request_summary["language"],
            request_summary["is_ocr"],
            request_summary["enable_table"],
            request_summary["enable_formula"],
            page_ranges or "all",
        )
        LOGGER.info("[MinerU-only] downstream disabled=%s", request_summary["downstream_disabled"])

        client = MinerUClient(
            token=token,
            api_base=request_summary["api_base"],
            model_version="vlm",
            language=request_summary["language"],
            enable_table=request_summary["enable_table"],
            enable_formula=request_summary["enable_formula"],
            is_ocr=request_summary["is_ocr"],
            poll_interval_seconds=float(mineru.get("poll_interval_seconds", 5)),
            timeout_seconds=effective_timeout_seconds,
        )
        result = client.parse_pdf(
            submission_pdf,
            output_dir,
            data_id=data_id,
            page_ranges=page_ranges,
        )
        archive_sha256 = _sha256(result.archive_path)
        same_as_baseline = bool(baseline_sha256 and archive_sha256 == baseline_sha256)
        layout_metadata = _layout_metadata(result.extracted_dir)
        returned_backend = str(layout_metadata.get("backend") or "").lower()
        backend_is_vlm = returned_backend == "vlm" or returned_backend.startswith("vlm-")
        returned_page_count = layout_metadata.get("page_count")
        full_document_requested = not page_ranges
        page_count_matches = (
            returned_page_count == source_page_count if full_document_requested else None
        )
        verification_errors: list[str] = []
        if same_as_baseline:
            verification_errors.append("returned ZIP is byte-identical to the rejected cached baseline")
        if not backend_is_vlm:
            verification_errors.append(
                f"returned layout backend is {returned_backend or 'missing'}, expected vlm"
            )
        if full_document_requested and not page_count_matches:
            verification_errors.append(
                f"returned page count is {returned_page_count!r}, expected {source_page_count}"
            )
        extracted_files = sorted(
            str(path.relative_to(output_dir))
            for path in result.extracted_dir.rglob("*")
            if path.is_file()
        )
        result_summary = {
            "status": "verification_failed" if verification_errors else "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "batch_id": result.batch_id,
            "model_version": "vlm",
            "archive_path": str(result.archive_path),
            "archive_bytes": result.archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "extracted_dir": str(result.extracted_dir),
            "extracted_file_count": len(extracted_files),
            "extracted_files": extracted_files,
            "mineru_result": _redact_result(result.raw_result),
            "mineru_layout_metadata": layout_metadata,
            "freshness_check": {
                "fresh_submission": fresh,
                "original_sha256": document_sha256,
                "submission_sha256": submission_sha256,
                "baseline_archive": str(baseline_path) if baseline_path else None,
                "baseline_sha256": baseline_sha256,
                "same_as_baseline": same_as_baseline,
            },
            "verification": {
                "requested_backend": "vlm",
                "returned_backend": returned_backend or None,
                "backend_is_vlm": backend_is_vlm,
                "full_document_requested": full_document_requested,
                "source_page_count": source_page_count,
                "returned_page_count": returned_page_count,
                "page_count_matches": page_count_matches,
                "errors": verification_errors,
            },
            "run_log": str(log_path),
            "downstream_executed": False,
        }
        verification_summary = result_summary
        _write_json(output_dir / "result.json", result_summary)
        if verification_errors:
            raise RuntimeError("MinerU result verification failed: " + "; ".join(verification_errors))
        LOGGER.info(
            "[MinerU-only] completed batch_id=%s archive=%s extracted_files=%s",
            result.batch_id,
            result.archive_path,
            len(extracted_files),
        )
        return result_summary
    except Exception as exc:
        failure = verification_summary or {"status": "failed", "run_log": str(log_path)}
        failure.update(
            {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(output_dir / "result.json", failure)
        LOGGER.exception("[MinerU-only] failed error=%s", failure)
        raise
    finally:
        if client is not None:
            client.close()
        package_logger.removeHandler(file_handler)
        file_handler.close()
