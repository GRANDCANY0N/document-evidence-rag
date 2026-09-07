from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


LOGGER = logging.getLogger(__name__)


class MinerUError(RuntimeError):
    """MinerU request, task, or archive failure."""


@dataclass(frozen=True)
class MinerUResult:
    batch_id: str
    full_zip_url: str
    extracted_dir: Path
    archive_path: Path
    raw_result: dict[str, Any]


class MinerUClient:
    """Client for MinerU v4 signed upload and batch result endpoints."""

    def __init__(
        self,
        token: str,
        api_base: str = "https://mineru.net",
        model_version: str = "vlm",
        language: str = "ch",
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = True,
        poll_interval_seconds: float = 5,
        timeout_seconds: float = 1800,
    ) -> None:
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.model_version = model_version
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.is_ocr = is_ocr
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(timeout=httpx.Timeout(60, read=120))
        self.download_client = httpx.Client(
            timeout=httpx.Timeout(60, read=180),
            follow_redirects=True,
            trust_env=False,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise MinerUError(f"MinerU API rejected request: {result.get('msg', 'unknown error')}")
        return result

    def request_upload(self, pdf_path: Path, data_id: str, page_ranges: str | None = None) -> tuple[str, str]:
        file_entry: dict[str, Any] = {
            "name": pdf_path.name,
            "data_id": data_id,
            # The v4 batch-upload API defines OCR at the per-file level.
            "is_ocr": self.is_ocr,
        }
        if page_ranges:
            file_entry["page_ranges"] = page_ranges
        payload = {
            "files": [file_entry],
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
        }
        result = self._post(f"{self.api_base}/api/v4/file-urls/batch", payload)
        data = result.get("data") or {}
        urls = data.get("file_urls") or []
        if not data.get("batch_id") or len(urls) != 1:
            raise MinerUError("MinerU did not return one signed upload URL")
        return str(data["batch_id"]), str(urls[0])

    def upload_file(self, pdf_path: Path, upload_url: str) -> None:
        with pdf_path.open("rb") as handle:
            response = self.client.put(upload_url, content=handle)
        if response.status_code not in (200, 201):
            raise MinerUError(f"MinerU signed upload failed with HTTP {response.status_code}")

    def _extract_result_items(self, response_data: dict[str, Any]) -> list[dict[str, Any]]:
        result = response_data.get("extract_result") or response_data.get("extract_results") or []
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []

    def wait_for_result(self, batch_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        url = f"{self.api_base}/api/v4/extract-results/batch/{batch_id}"
        poll_count = 0
        previous_states: tuple[str, ...] = ()
        while time.monotonic() < deadline:
            poll_count += 1
            response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise MinerUError(f"MinerU result query failed: {payload.get('msg', 'unknown error')}")
            items = self._extract_result_items(payload.get("data") or {})
            states = tuple(str(item.get("state") or "unknown") for item in items)
            if states != previous_states or poll_count == 1 or poll_count % 12 == 0:
                LOGGER.info(
                    "[MinerU] poll batch_id=%s poll=%s states=%s",
                    batch_id,
                    poll_count,
                    list(states) or ["waiting"],
                )
                previous_states = states
            for item in items:
                state = item.get("state")
                if state == "done" and item.get("full_zip_url"):
                    return item
                if state == "failed":
                    raise MinerUError(f"MinerU extraction failed: {item.get('err_msg', 'unknown error')}")
            time.sleep(self.poll_interval_seconds)
        raise MinerUError(f"MinerU extraction timed out after {self.timeout_seconds}s")

    def download_and_extract(self, full_zip_url: str, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "mineru-result.zip"
        archive_bytes = self._download_bytes(full_zip_url)
        archive_path.write_bytes(archive_bytes)
        LOGGER.info("[MinerU] archive downloaded bytes=%s path=%s", len(archive_bytes), archive_path)
        extracted_dir = output_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            root = extracted_dir.resolve()
            for member in archive.infolist():
                target = (extracted_dir / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise MinerUError(f"Unsafe path in MinerU archive: {member.filename}")
            archive.extractall(extracted_dir)
            LOGGER.info("[MinerU] archive extracted files=%s path=%s", len(archive.infolist()), extracted_dir)
        return archive_path, extracted_dir

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _download_bytes(self, url: str) -> bytes:
        response = self.download_client.get(url)
        response.raise_for_status()
        return response.content

    def parse_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        data_id: str,
        page_ranges: str | None = None,
    ) -> MinerUResult:
        batch_id, upload_url = self.request_upload(pdf_path, data_id, page_ranges=page_ranges)
        LOGGER.info("[MinerU] upload URL acquired batch_id=%s", batch_id)
        self.upload_file(pdf_path, upload_url)
        LOGGER.info("[MinerU] upload completed batch_id=%s bytes=%s", batch_id, pdf_path.stat().st_size)
        raw_result = self.wait_for_result(batch_id)
        archive_path, extracted_dir = self.download_and_extract(str(raw_result["full_zip_url"]), output_dir)
        return MinerUResult(
            batch_id=batch_id,
            full_zip_url=str(raw_result["full_zip_url"]),
            extracted_dir=extracted_dir,
            archive_path=archive_path,
            raw_result=raw_result,
        )

    def close(self) -> None:
        self.client.close()
        self.download_client.close()
