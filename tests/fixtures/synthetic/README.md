# Synthetic PDF fixtures

This directory contains deterministic, fully synthetic PDF fixtures for document layout, OCR, table, chart, formula, and anti-hallucination tests. No external API, database, project configuration, or real user document is used.

## Generate

Requirements:

- Python 3.10+
- Pillow
- one of the font files listed in `generate_fixtures.py`
- optional: `pdfinfo` for an independent structural/page-count check

From the repository root:

```bash
python3 tests/fixtures/synthetic/generate_fixtures.py
```

The command overwrites only generated PDFs, `manifest.json`, and `ground_truth.json` in this directory. It validates the PDF header/EOF marker and page objects; when available, it also validates every page count with `pdfinfo`.

The PDF writer omits timestamps and uses fixed object ordering. Given the same Pillow/font versions, repeated runs produce the same PDF SHA-256 values recorded in `manifest.json`.

## Covered scenarios

| PDF | Coverage |
| --- | --- |
| `two_column_reading_order.pdf` | Two-column region detection and non-interleaved reading order |
| `mixed_orientation.pdf` | Horizontal plus vertical text, rotated text, and portrait/landscape pages in one PDF |
| `cross_page_table_boundary_surrogate.pdf` | Cross-page rows, repeated headers, and a compact logical page 62–66 surrogate around the page-64 boundary |
| `borderless_multilevel_merged_tables.pdf` | Borderless table, multi-level/column-spanning header, and row-spanning merged cells |
| `degraded_images.pdf` | Sharp control, Gaussian blur, and low-contrast raster text |
| `flowchart_statistical_chart.pdf` | Directed decision flowchart and grouped-bar/line statistical chart |
| `complex_formulas.pdf` | Integral, limit, sum, fraction, matrix, radicals, Greek symbols, and a PDE |
| `headers_footers_watermarks.pdf` | Repeated running headers/footers, page numbering, watermark classification, and unique body content |
| `occlusion_irrecoverable.pdf` | Fully blacked-out, explicitly irrecoverable fields and anti-hallucination behavior |

Exact expected tokens, table shapes, graph edges, chart values, normalized formulas, and recoverability labels are in `ground_truth.json`. Generated file sizes, hashes, page counts, font paths, and validators are in `manifest.json`.

## Compact page-64 boundary surrogate

The default cross-page fixture has five physical pages mapped to logical pages 62, 63, 64, 65, and 66. Physical page 3 is therefore the logical page-64 boundary. This small fixture is suitable for regular CI while still testing boundary arithmetic, repeated-header handling, and row sequence continuity.

For an integration/stress run that needs a real physical page 64, generate the optional 65-page artifact:

```bash
python3 tests/fixtures/synthetic/generate_fixtures.py --include-full-65-page-boundary
```

That adds `cross_page_table_full_65_pages.pdf`; it is intentionally not part of the default checked-in fixture set because it is much larger. Running the default command again leaves that optional file untouched, so remove it explicitly if a test created it in a disposable output directory.

## Design notes

- All PDF pages are image-only at 144 DPI. This makes the fixtures suitable for VLM/OCR pipelines and guarantees that occluded fields have no hidden PDF text layer.
- The occlusion fixture never draws a source value before covering the field. Ground truth uses `[OCCLUDED]` and marks the value non-recoverable; guessing a value is a failure.
- Content is synthetic and uses stable identifiers to make reading-order and extraction assertions exact.
- To avoid changing this source directory during experiments, pass `--output-dir /path/to/temp-dir`.
