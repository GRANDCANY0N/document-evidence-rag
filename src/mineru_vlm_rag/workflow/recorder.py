from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mineru_vlm_rag.domain.models import WorkflowEvent, new_id


class WorkflowRecorder:
    """Records explainable per-document decisions as JSONL and in-memory events."""

    def __init__(self, document_id: str, output_dir: Path) -> None:
        self.document_id = document_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / f"{document_id}.workflow.jsonl"
        self.run_id = new_id()
        self.events: list[WorkflowEvent] = []

    def record(
        self,
        stage: str,
        action: str,
        decision: str,
        *,
        page_no: int | None = None,
        block_id: str | None = None,
        issue_flags: list[str] | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        event = WorkflowEvent(
            document_id=self.document_id,
            stage=stage,
            action=action,
            decision=decision,
            page_no=page_no,
            block_id=block_id,
            issue_flags=issue_flags or [],
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            details={"run_id": self.run_id, **(details or {})},
        )
        self.events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def write_markdown_summary(self) -> Path:
        path = self.output_dir / f"{self.document_id}.workflow.md"
        lines = [
            f"# 文档复杂问题处理记录 `{self.document_id}`",
            "",
            f"运行 ID：`{self.run_id}`",
            "",
            "| 时间 | 阶段 | 页码 | Block | 问题码 | 动作 | 决策 |",
            "|---|---|---:|---|---|---|---|",
        ]
        for event in self.events:
            flags = ", ".join(event.issue_flags) or "-"
            lines.append(
                f"| {event.created_at.isoformat()} | {event.stage} | {event.page_no or '-'} | "
                f"{event.block_id or '-'} | {flags} | {event.action} | {event.decision} |"
            )
            if event.details:
                lines.extend(["", "```json", json.dumps(event.details, ensure_ascii=False, indent=2), "```", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
