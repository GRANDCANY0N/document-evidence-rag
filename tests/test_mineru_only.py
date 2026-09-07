import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

import mineru_vlm_rag.mineru_only as mineru_only_module
from mineru_vlm_rag.adapters.mineru_client import MinerUClient, MinerUResult
from mineru_vlm_rag.cli import _parser
from mineru_vlm_rag.mineru_only import _create_fresh_submission_copy, run_mineru_vlm_only


def test_batch_upload_places_ocr_on_file_entry(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    captured = {}
    client = MinerUClient(token="test-token", model_version="vlm", is_ocr=True)

    def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"data": {"batch_id": "batch-1", "file_urls": ["https://upload.invalid/file"]}}

    client._post = fake_post  # type: ignore[method-assign]
    try:
        batch_id, upload_url = client.request_upload(pdf, "document-1", page_ranges="1-3")
    finally:
        client.close()

    assert batch_id == "batch-1"
    assert upload_url == "https://upload.invalid/file"
    assert captured["payload"]["model_version"] == "vlm"
    assert "is_ocr" not in captured["payload"]
    assert captured["payload"]["files"] == [
        {
            "name": "sample.pdf",
            "data_id": "document-1",
            "is_ocr": True,
            "page_ranges": "1-3",
        }
    ]


def test_mineru_only_cli_arguments() -> None:
    args = _parser().parse_args(
        [
            "--verbose",
            "mineru-only",
            "/tmp/input.pdf",
            "--output-dir",
            "/tmp/mineru-output",
            "--page-ranges",
            "1-10",
            "--fresh",
            "--reject-same-as",
            "/tmp/baseline.zip",
            "--timeout-seconds",
            "7200",
        ]
    )
    assert args.command == "mineru-only"
    assert args.verbose is True
    assert args.pdf == Path("/tmp/input.pdf")
    assert args.output_dir == Path("/tmp/mineru-output")
    assert args.page_ranges == "1-10"
    assert args.fresh is True
    assert args.reject_same_as == Path("/tmp/baseline.zip")
    assert args.timeout_seconds == 7200


def test_fresh_submission_changes_bytes_but_not_pdf_pages(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as handle:
        writer.write(handle)

    submission, nonce = _create_fresh_submission_copy(source, tmp_path)

    assert submission != source
    assert submission.read_bytes().startswith(source.read_bytes())
    assert submission.read_bytes() != source.read_bytes()
    assert f"MinerU fresh-run nonce: {nonce}".encode("ascii") in submission.read_bytes()
    assert len(PdfReader(str(source)).pages) == 2
    assert len(PdfReader(str(submission)).pages) == 2


def _write_two_page_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _write_test_settings(tmp_path: Path) -> tuple[Path, Path]:
    env_path = tmp_path / ".env"
    env_path.write_text("MINERU_API_TOKEN=test-token\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mineru:\n  api_base: https://example.invalid\n  timeout_seconds: 30\n",
        encoding="utf-8",
    )
    return env_path, config_path


def _fake_client_class(backend: str):
    class FakeMinerUClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def parse_pdf(self, pdf_path, output_dir, data_id, page_ranges=None) -> MinerUResult:
            archive_path = output_dir / "mineru-result.zip"
            archive_path.write_bytes(f"new-{backend}".encode("ascii"))
            extracted_dir = output_dir / "extracted"
            extracted_dir.mkdir()
            (extracted_dir / "layout.json").write_text(
                json.dumps({"_backend": backend, "pdf_info": [{}, {}]}),
                encoding="utf-8",
            )
            return MinerUResult(
                batch_id="batch-test",
                full_zip_url="https://example.invalid/result.zip?secret=redacted",
                extracted_dir=extracted_dir,
                archive_path=archive_path,
                raw_result={"state": "done", "full_zip_url": self.full_zip_url},
            )

        full_zip_url = "https://example.invalid/result.zip?secret=redacted"

        def close(self) -> None:
            return None

    return FakeMinerUClient


def test_full_vlm_run_verifies_backend_pages_and_freshness(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    env_path, config_path = _write_test_settings(tmp_path)
    baseline = tmp_path / "baseline.zip"
    baseline.write_bytes(b"old-hybrid")
    output_dir = tmp_path / "run"
    monkeypatch.setattr(mineru_only_module, "MinerUClient", _fake_client_class("vlm"))

    result = run_mineru_vlm_only(
        source,
        output_dir,
        fresh=True,
        reject_same_as=baseline,
        env_path=env_path,
        config_path=config_path,
    )

    assert result["status"] == "completed"
    assert result["verification"]["backend_is_vlm"] is True
    assert result["verification"]["page_count_matches"] is True
    assert result["freshness_check"]["same_as_baseline"] is False


def test_full_vlm_run_rejects_hybrid_result(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    env_path, config_path = _write_test_settings(tmp_path)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(mineru_only_module, "MinerUClient", _fake_client_class("hybrid"))

    with pytest.raises(RuntimeError, match="backend is hybrid"):
        run_mineru_vlm_only(
            source,
            output_dir,
            fresh=True,
            env_path=env_path,
            config_path=config_path,
        )

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "verification_failed"
    assert result["verification"]["backend_is_vlm"] is False
