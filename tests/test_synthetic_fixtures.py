from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mineru_vlm_rag.pdf import inspect_pdf


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic"


def test_synthetic_fixture_manifest_is_complete_and_valid() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((FIXTURE_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    expected_pages = {item["file"]: item["page_count"] for item in ground_truth["documents"]}
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 9
    assert sum(item["page_count"] for item in artifacts) == 19
    for artifact in artifacts:
        path = FIXTURE_DIR / artifact["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        preflight = inspect_pdf(path)
        assert digest == artifact["sha256"]
        assert preflight.page_count == artifact["page_count"] == expected_pages[path.name]


def test_ground_truth_covers_required_complex_scenarios() -> None:
    payload = json.loads((FIXTURE_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    scenarios = {
        scenario
        for document in payload["documents"]
        for scenario in document["scenario_ids"]
    }
    required = {
        "two_column_reading_order",
        "horizontal_vertical_text",
        "cross_page_table",
        "logical_page_64_boundary",
        "flowchart",
        "statistical_chart",
        "complex_formula",
        "running_header",
        "running_footer",
        "occlusion",
        "irrecoverable_field",
    }
    assert required <= scenarios
