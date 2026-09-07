from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, TypeVar

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


T = TypeVar("T", bound=BaseModel)


class VLMResult(BaseModel):
    task: str
    readable: bool = True
    text: str = ""
    html: str = ""
    latex: str = ""
    summary: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    uncertainty: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    is_continuation: bool | None = None
    column_mapping: list[dict[str, Any]] = Field(default_factory=list)


class TableStructureResponse(BaseModel):
    readable: bool = True
    rotation_degrees: int = 0
    header_region: list[float] = Field(default_factory=list)
    data_region: list[float] = Field(default_factory=list)
    header_rows: int = 1
    column_count: int = 0
    title: str = ""
    date: str = ""
    unit: str = ""
    footnotes: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class TableCellsResponse(BaseModel):
    readable: bool = True
    html: str = ""
    row_count: int = 0
    column_count: int = 0
    header_rows: int = 1
    uncertainty: list[str] = Field(default_factory=list)


PROMPTS: dict[str, str] = {
    "ocr_verify": "逐字识别可见文字，保留大小写、编号、金额、日期、标点和换行。不可读处不要猜测。",
    "table_to_cells": (
        "逐单元格识别表格并输出合法HTML。保留表标题、日期、单位、脚注、多级表头、rowspan、colspan、"
        "负号、小数位、百分号和公式；不得四舍五入或根据上下文补数。HTML是单元格事实的唯一完整副本，"
        "不要在structured_data中重复全部cells，只返回row_count、column_count和header_rows，以免输出截断。"
        "不可读单元格写入uncertainty，不要猜测。"
    ),
    "table_structure": (
        "本次只分析表格结构，不抄录全部数据。识别表标题、日期、单位、脚注、多层表头层数、列数，"
        "rotation_degrees只能是0、90、180、270，表示为了让文字正常横向阅读需要顺时针旋转的角度。"
        "header_region和data_region必须以完成上述旋转后的图像为坐标系，返回0到1归一化的"
        "header_region=[x0,y0,x1,y1]、data_region=[x0,y0,x1,y1]，以及"
        "header_rows、column_count、title、date、unit、footnotes。"
        "边界不确定时写入uncertainty，不得虚构。"
    ),
    "visual_analyze": (
        "先判断图像的真实类型，只能是table、chart、diagram、photo、seal或other，并把结果放入"
        "structured_data.visual_type。若为表格，只分析表标题、日期、单位、脚注、多层表头、数据区边界，"
        "不要在本次抄录全部数据；将0到1归一化的header_region和data_region写入structured_data。"
        "若不是表格，在同一次请求中完成图表/流程图/图片事实提取，保留可见文字和数值，不得猜测。"
    ),
    "table_continuation": "判断多张相邻页表格图是否属于同一张跨页表，并给出列对应关系。只做判断，不补造单元格。",
    "chart_extract": "提取图表标题、坐标轴、单位、图例、可见数据点和趋势。看不清的数值不要猜测。",
    "diagram_extract": "提取流程图或关系图的标题、节点、方向边、可见文字与关系；不得添加图中不存在的节点。",
    "formula_to_latex": "将可见公式转写为LaTeX并识别公式编号。不可读部分明确标记。",
    "reading_order": "根据整页图和候选块编号，只返回正确阅读顺序的块编号列表，不重写正文。",
    "seal_or_stamp": "识别印章或签章中清晰可见的文字、形状和日期；模糊或遮挡部分不要猜测。",
}


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


class SiliconFlowClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        vlm_model: str,
        embedding_model: str,
        rerank_model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        image_detail: str = "high",
        max_retries: int = 1,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.vlm_model = vlm_model
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_detail = image_detail
        self.max_retries = max(0, max_retries)
        # SiliconFlow is directly reachable from this server.  Ignoring proxy
        # environment variables here is a final guard in case the launching
        # shell has not reloaded NO_PROXY yet.  SDK network retries are disabled
        # so a 5-minute timeout cannot silently expand into 15 minutes.
        openai_http = httpx.Client(timeout=300, trust_env=False)
        self.openai = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=300,
            max_retries=0,
            http_client=openai_http,
        )
        self.http = httpx.Client(
            timeout=300,
            trust_env=False,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def analyze_images(
        self,
        image_paths: list[Path],
        task: str,
        context: str = "",
    ) -> VLMResult:
        if task not in PROMPTS:
            raise ValueError(f"Unsupported VLM task: {task}")
        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(path), "detail": self.image_detail},
                }
            )
        response_model: type[BaseModel]
        if task == "table_structure":
            response_model = TableStructureResponse
        elif task == "table_to_cells":
            response_model = TableCellsResponse
        else:
            response_model = VLMResult
        schema_hint = response_model.model_json_schema()
        prompt = (
            f"任务类型：{task}\n{PROMPTS[task]}\n上下文：{context or '无'}\n"
            "严格返回单个JSON对象，不要使用Markdown代码围栏。"
            f"输出字段必须符合此JSON Schema：{json.dumps(schema_hint, ensure_ascii=False)}"
        )
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            response = self.openai.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            try:
                payload = _json_from_text(raw)
                if response_model is VLMResult:
                    payload.setdefault("task", task)
                    return VLMResult.model_validate(payload)
                if response_model is TableStructureResponse:
                    value = TableStructureResponse.model_validate(payload)
                    rotation = value.rotation_degrees if value.rotation_degrees in {0, 90, 180, 270} else 0
                    return VLMResult(
                        task=task,
                        readable=value.readable,
                        structured_data={
                            "rotation_degrees": rotation,
                            "header_region": value.header_region,
                            "data_region": value.data_region,
                            "header_rows": value.header_rows,
                            "column_count": value.column_count,
                            "title": value.title,
                            "date": value.date,
                            "unit": value.unit,
                            "footnotes": value.footnotes,
                        },
                        uncertainty=value.uncertainty,
                    )
                value = TableCellsResponse.model_validate(payload)
                return VLMResult(
                    task=task,
                    readable=value.readable,
                    html=value.html,
                    structured_data={
                        "row_count": value.row_count,
                        "column_count": value.column_count,
                        "header_rows": value.header_rows,
                    },
                    uncertainty=value.uncertainty,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    messages.extend([
                        {"role": "assistant", "content": raw[:12000]},
                        {
                            "role": "user",
                            "content": "上次输出未通过JSON Schema。请纠正字段类型并只返回一个合法JSON对象。",
                        },
                    ])
        assert last_error is not None
        raise ValueError(f"Invalid structured VLM output for {task}: {type(last_error).__name__}") from last_error

    def embed(self, texts: list[str], dimensions: int = 0) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.embedding_model, "input": texts, "encoding_format": "float"}
        if dimensions > 0:
            kwargs["dimensions"] = dimensions
        response = self.openai.embeddings.create(**kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        if not documents:
            return []
        response = self.http.post(
            f"{self.base_url}/rerank",
            json={
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
                "instruction": "按与查询的事实相关性、表格字段匹配和证据完整性重排序。",
            },
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])

    def close(self) -> None:
        self.openai.close()
        self.http.close()
