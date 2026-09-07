# DocTrace · Multimodal Retrieval for Complex Documents

面向扫描件、多栏排版、复杂表格、图表与公式 PDF 的证据检索工程。系统从文档预检开始，依次完成 MinerU 解析、质量门控、Qwen3-VL 按需修订、结构重建、多模态切块、存证与检索；召回结果保留页码、坐标、资产和修订来源，便于回到原文复核。

> 当前仓库聚焦“解析到证据召回”，不包含最终答案生成器。真实文档、模型产物、数据库、日志和密钥均未纳入公开仓库。

## Pipeline

```text
PDF preflight
  → MinerU page/block/asset extraction
  → completeness & quality audit
  → targeted Qwen3-VL revision
  → reading order & cross-page table reconstruction
  → multimodal chunking
  → MySQL evidence store
  → Milvus retrieval & reranking
```

## Engineering highlights

- **非破坏式修订**：原始解析结果与 VLM 修订分层保存；冲突内容标记为待复核，不静默覆盖。
- **按需视觉路由**：先由确定性规则定位模糊、乱码、旋转和结构异常，再按 block 类型调用专用视觉任务。
- **复杂表格处理**：支持多级表头、按真实行边界切片、重复表头、跨页表判断与 canonical cell 对比。
- **可追溯 Chunk**：正文、表格、图表和公式采用不同粒度，并携带 page、bbox、asset 与 evidence status。
- **可回归评测**：包含合成复杂 PDF fixture、结构质量测试与代表性召回问题集。

## Project structure

```text
src/mineru_vlm_rag/   核心实现
config/                解析、质量与切块参数
tests/                 自动化测试与合成 PDF fixture
evaluation/            召回评测问题及说明
scripts/               报告生成和定向重试工具
```

## Quick start

Requires Python 3.11+ and locally available MySQL. External MinerU and SiliconFlow services are configured through environment variables.

```bash
git clone https://github.com/GRANDCANY0N/document-evidence-rag.git
cd document-evidence-rag
python -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e '.[dev]'
cp .env.example .env
```

Fill in the credentials in `.env`, then run:

```bash
mineru-vlm-rag init-db
mineru-vlm-rag doctor
mineru-vlm-rag ingest /path/to/document.pdf
mineru-vlm-rag query '查询内容' --document-id <document_id> --limit 8
```

Run tests:

```bash
python -m pytest -q
```

## Security and data policy

This public repository intentionally excludes `.env`, internal design documents, source documents, parsed outputs, vector databases, runtime logs and production identifiers. Use only documents and service credentials you are authorized to process.
