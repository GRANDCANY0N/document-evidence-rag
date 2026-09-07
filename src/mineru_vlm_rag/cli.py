from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from sqlalchemy import create_engine, text

from mineru_vlm_rag.chunking import build_chunks, write_chunk_export, write_linear_export
from mineru_vlm_rag.adapters.siliconflow import SiliconFlowClient
from mineru_vlm_rag.domain.models import Chunk
from mineru_vlm_rag.evaluation import audit_chunk_export, evaluate_retrieval, write_chunk_quality_report
from mineru_vlm_rag.mineru_only import run_mineru_vlm_only
from mineru_vlm_rag.normalization import normalize_document_structure
from mineru_vlm_rag.pipeline import IngestionPipeline, QueryService
from mineru_vlm_rag.persistence import MilvusRepository, MySQLRepository
from mineru_vlm_rag.settings import load_settings
from mineru_vlm_rag.tables import revert_unsafe_table_merges


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mineru-vlm-rag")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    sub.add_parser("doctor")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--no-vlm", action="store_true")
    ingest.add_argument(
        "--fresh-mineru",
        action="store_true",
        help="submit a byte-distinct, visually identical PDF so MinerU does not reuse an input-hash cache",
    )
    ingest.add_argument(
        "--reuse-mineru",
        action="store_true",
        help="reuse outputs/<document-id>/mineru from a completed upload and rerun downstream stages",
    )
    rechunk = sub.add_parser(
        "rechunk",
        help="rebuild multimodal chunks from the MySQL block graph and export JSON; never call embedding or Milvus",
    )
    rechunk.add_argument("document_id")
    rechunk.add_argument("--output", type=Path)
    rechunk.add_argument("--text-size", type=int)
    rechunk.add_argument("--overlap", type=int)
    rechunk.add_argument("--table-row-group-size", type=int)
    audit_chunks = sub.add_parser(
        "audit-chunks",
        help="run deterministic quality gates on an exported all_chunks.json",
    )
    audit_chunks.add_argument("input", type=Path)
    audit_chunks.add_argument("--output-dir", type=Path)
    index_chunks = sub.add_parser(
        "index-chunks",
        help="quality-gate, embed and persist an exported all_chunks.json without rerunning MinerU/VLM",
    )
    index_chunks.add_argument("document_id")
    index_chunks.add_argument("--input", type=Path, required=True)
    index_chunks.add_argument("--report-dir", type=Path)
    eval_retrieval = sub.add_parser(
        "eval-retrieval",
        help="run PDF-grounded questions against the actual embedding+rerank index",
    )
    eval_retrieval.add_argument("document_id")
    eval_retrieval.add_argument("--questions", type=Path, required=True)
    eval_retrieval.add_argument("--output-dir", type=Path, required=True)
    eval_retrieval.add_argument("--limit", type=int, default=10)
    mineru_only = sub.add_parser(
        "mineru-only",
        help="run MinerU's remote VLM parser only; skip local VLM, DB, chunking and embedding",
    )
    mineru_only.add_argument("pdf", type=Path)
    mineru_only.add_argument("--output-dir", type=Path, required=True)
    mineru_only.add_argument("--page-ranges")
    mineru_only.add_argument(
        "--fresh",
        action="store_true",
        help="submit a visually identical byte-distinct PDF copy to avoid input-hash cache reuse",
    )
    mineru_only.add_argument(
        "--reject-same-as",
        type=Path,
        help="fail if the returned MinerU ZIP is byte-identical to this baseline archive",
    )
    mineru_only.add_argument(
        "--timeout-seconds",
        type=float,
        help="maximum time to wait for the MinerU batch result (default: config value)",
    )
    query = sub.add_parser("query")
    query.add_argument("text")
    query.add_argument("--document-id")
    query.add_argument("--block-type")
    query.add_argument("--limit", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    if args.command == "mineru-only":
        result = run_mineru_vlm_only(
            args.pdf,
            args.output_dir,
            page_ranges=args.page_ranges,
            fresh=args.fresh,
            reject_same_as=args.reject_same_as,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "audit-chunks":
        input_path = args.input.resolve()
        output_dir = (args.output_dir or input_path.parent / "quality").resolve()
        result = audit_chunk_export(input_path)
        write_chunk_quality_report(
            result,
            output_dir / "chunk_quality_report.json",
            output_dir / "chunk_quality_report.md",
        )
        print(json.dumps({
            **result,
            "json_report": str(output_dir / "chunk_quality_report.json"),
            "markdown_report": str(output_dir / "chunk_quality_report.md"),
        }, ensure_ascii=False, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)
        return
    settings = load_settings()
    if args.command == "index-chunks":
        input_path = args.input.resolve()
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        document_id = str(payload.get("document", {}).get("document_id") or "")
        if document_id != args.document_id:
            raise ValueError(
                f"Chunk export document_id={document_id!r} does not match {args.document_id!r}"
            )
        audit = audit_chunk_export(input_path)
        if audit["status"] != "passed":
            raise ValueError(f"Chunk quality gate failed: {audit['hard_failures']}")
        chunks = [Chunk.model_validate(value) for value in payload.get("chunks") or []]
        if not chunks:
            raise ValueError("Chunk export contains no chunks")
        report_dir = (args.report_dir or input_path.parent / "indexing").resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(settings.pipeline.get("milvus_collection_prefix", "document_chunks_v3"))
        batch_size = int(settings.embedding["batch_size"])
        silicon = SiliconFlowClient(
            api_key=settings.siliconflow_api_key,
            base_url=settings.embedding["api_base"],
            vlm_model=settings.vlm_model,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            max_retries=0,
        )
        milvus = MilvusRepository(settings.milvus_uri, collection_prefix=prefix)
        mysql = MySQLRepository(settings.mysql_dsn)
        started_at = time.monotonic()
        vectors: list[list[float]] = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        batch_durations: list[float] = []
        try:
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset:offset + batch_size]
                batch_index = offset // batch_size + 1
                logging.info(
                    "[离线Embedding] start batch=%s/%s chunks=%s-%s count=%s",
                    batch_index, total_batches, offset + 1, offset + len(batch), len(batch),
                )
                batch_started = time.monotonic()
                values = silicon.embed(
                    [chunk.embedding_text for chunk in batch],
                    dimensions=int(settings.embedding["dimensions"]),
                )
                if len(values) != len(batch):
                    raise ValueError(
                        f"Embedding batch {batch_index} returned {len(values)} vectors for {len(batch)} chunks"
                    )
                duration = round(time.monotonic() - batch_started, 3)
                batch_durations.append(duration)
                vectors.extend(values)
                logging.info(
                    "[离线Embedding] done batch=%s/%s duration=%.3fs dimension=%s",
                    batch_index, total_batches, duration, len(values[0]) if values else 0,
                )
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise ValueError("Embedding dimensions are inconsistent")
            logging.info("[Milvus-v3] upsert start collection_prefix=%s chunks=%s", prefix, len(chunks))
            collection = milvus.upsert(chunks, vectors, replace_documents=True)
            logging.info("[Milvus-v3] upsert done collection=%s", collection)
            mysql_count = mysql.replace_document_chunks(
                args.document_id,
                chunks,
                embedding_state="embedded",
            )
            elapsed = round(time.monotonic() - started_at, 3)
            result = {
                "status": "ok",
                "document_id": args.document_id,
                "input": str(input_path),
                "chunk_schema_version": payload.get("schema_version"),
                "chunk_quality_score": audit.get("score"),
                "chunk_count": len(chunks),
                "embedding_model": settings.embedding_model,
                "embedding_dimension": dimension,
                "embedding_batch_size": batch_size,
                "embedding_batch_count": total_batches,
                "embedding_batch_durations_seconds": batch_durations,
                "milvus_collection": collection,
                "mysql_chunk_count": mysql_count,
                "elapsed_seconds": elapsed,
                "mineru_or_vlm_rerun": False,
            }
            report_path = report_dir / "indexing_report.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({**result, "report": str(report_path)}, ensure_ascii=False, indent=2))
        finally:
            silicon.close()
            milvus.close()
            mysql.engine.dispose()
        return
    if args.command == "init-db":
        pipeline = IngestionPipeline(settings)
        try:
            pipeline.initialize()
            print(json.dumps({"status": "ok", "database": "initialized"}, ensure_ascii=False))
        finally:
            pipeline.close()
        return
    if args.command == "doctor":
        engine = create_engine(settings.mysql_dsn)
        with engine.connect() as connection:
            mysql_ok = connection.execute(text("SELECT 1")).scalar() == 1
        milvus = MilvusRepository(
            settings.milvus_uri,
            collection_prefix=str(settings.pipeline.get("milvus_collection_prefix", "document_chunks_v3")),
        )
        milvus.close()
        print(json.dumps({"mysql": mysql_ok, "milvus": True, "settings": True}, ensure_ascii=False))
        return
    if args.command == "rechunk":
        repository = MySQLRepository(settings.mysql_dsn)
        try:
            document = repository.load_document_graph(args.document_id)
            table_repair = revert_unsafe_table_merges(document)
            structure_summary = normalize_document_structure(document)
            text_size = (
                args.text_size
                if args.text_size is not None
                else int(settings.pipeline["text_chunk_chars"])
            )
            overlap = args.overlap if args.overlap is not None else int(settings.pipeline["text_chunk_overlap"])
            row_group_size = (
                args.table_row_group_size
                if args.table_row_group_size is not None
                else int(settings.pipeline.get("table_row_group_size", 6))
            )
            if text_size <= 0 or overlap < 0 or overlap >= text_size or row_group_size <= 0:
                raise ValueError(
                    "text-size must be positive, overlap must be in [0, text-size), "
                    "and table-row-group-size must be positive"
                )
            chunks = build_chunks(
                document,
                text_size=text_size,
                overlap=overlap,
                table_row_group_size=row_group_size,
            )
            output_path = args.output or (
                settings.project_root / "outputs" / args.document_id / "chunks" / "all_chunks.json"
            )
            parameters = {
                "text_size_chars": text_size,
                "text_overlap_chars": overlap,
                "text_length_unit": "unicode_code_points",
                "parameter_status": "provisional_requires_rag_evaluation",
                "table_row_group_size": row_group_size,
                "unresolved_content_policy": "mineru_raw_as_provisional",
            }
            payload = write_chunk_export(document, chunks, output_path, parameters=parameters)
            linear_output_path = output_path.parent / "linear_units.json"
            linear_payload = write_linear_export(document, linear_output_path)
            print(json.dumps({
                "status": "ok",
                "document_id": args.document_id,
                "output": str(output_path.resolve()),
                "embedding_executed": False,
                "database_chunks_replaced": False,
                "linear_output": str(linear_output_path.resolve()),
                "linear_unit_count": linear_payload["unit_count"],
                "table_merge_repair": table_repair,
                "structure_normalization": structure_summary,
                **payload["summary"],
            }, ensure_ascii=False, indent=2))
        finally:
            repository.engine.dispose()
        return
    if args.command == "ingest":
        pipeline = IngestionPipeline(settings)
        try:
            pipeline.initialize()
            result = pipeline.ingest(
                args.pdf,
                use_vlm=not args.no_vlm,
                fresh_mineru=args.fresh_mineru,
                reuse_mineru=args.reuse_mineru,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            pipeline.close()
        return
    if args.command == "query":
        service = QueryService(settings)
        try:
            result = service.search(
                args.text,
                document_id=args.document_id,
                block_type=args.block_type,
                limit=args.limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            service.close()
        return
    if args.command == "eval-retrieval":
        service = QueryService(settings)
        try:
            result = evaluate_retrieval(
                args.questions.resolve(),
                service.search,
                document_id=args.document_id,
                output_dir=args.output_dir.resolve(),
                limit=args.limit,
            )
            print(json.dumps({
                "status": result["status"],
                "document_id": result["document_id"],
                "question_count": result["question_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "summary": result["summary"],
                "json_report": str((args.output_dir / "retrieval_quality_report.json").resolve()),
                "markdown_report": str((args.output_dir / "retrieval_quality_report.md").resolve()),
            }, ensure_ascii=False, indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    main()
