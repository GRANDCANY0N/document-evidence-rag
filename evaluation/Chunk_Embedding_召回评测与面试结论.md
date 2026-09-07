# Chunk、Embedding 与召回评测结论

## 1. 本次结论

评测文档为 `input/2025金融稳定报告.pdf`，文档 ID 为：

`4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393`

本次已经完成以下闭环：

1. 对最终 Chunk JSON 做确定性结构质量门控；
2. 使用真实 Embedding API 生成 2560 维向量；
3. 将同一批 1246 个 Chunk 写入 MySQL 和 Milvus；
4. 由独立 subagent 只阅读原 PDF 构造 48 个问题，不读取 Chunk、数据库或召回结果；
5. 对正文、表格、图表、脚注和跨页内容执行真实召回与重排评测；
6. 根据失败样例修复跨页重复片段、阅读顺序和多行表格证据选择，再做全量回归。

最终结果不是“MinerU 字符识别准确率”，而是这份 PDF 在当前索引与问题集上的**证据召回结果**。Top3 的目标页命中、内容命中、联合命中、严格关键词命中和平均关键词覆盖均为 100%。

## 2. Chunk 质量门控

质量报告得分为 **99/100，状态 passed，无硬失败**。

| 项目 | 结果 |
| --- | ---: |
| Chunk 总数 | 1246 |
| 逻辑父节点数 | 483 |
| verified | 1120 |
| provisional | 126 |
| 完全重复组 | 0 |
| 页码起点倒退 | 0 |
| 页码终点倒退 | 0 |
| 非法表头映射 | 0 |
| 短文本警告 | 38 |

38 个短文本经检查主要是“目录”“第一章 宏观经济运行”“数据来源”等标题或来源标签，不是被截断的正文，所以保留为可检索结构信号。它们是 warning，不是 hard failure。

各模态 Chunk 数量：

| 模态 | 数量 |
| --- | ---: |
| 正文 | 329 |
| 表格 | 773 |
| 图表 | 125 |
| 脚注 | 19 |

各角色数量：`text_segment=329`、`table_summary=33`、`table_row_group=110`、`table_row=628`、`table_fallback=2`、`chart_summary=37`、`chart_fact=88`、`footnote=19`。

本轮重点解决了三个会传播到 Chunk 的结构问题：

- 识别并标记 40 个跨页顶部的完全重复承接片段；源 block 仍保留，只设置 `skip_chunk`，避免索引重复内容。
- 修复第 29 页“截至2024年末”被双栏顺序拆开的局部阅读顺序。
- 将跨正文区、宽度达到页面可用区 58% 的标题作为双栏分隔锚点，恢复第 20—21 页“超过1900家专精特新企业在A股上市”的连续正文。

## 3. Embedding 与入库结果

| 项目 | 实际结果 |
| --- | --- |
| Embedding 模型 | `Qwen/Qwen3-Embedding-4B` |
| 向量维度 | 2560 |
| 批大小 | 16 |
| 批次数 | 78 |
| 总耗时 | 71.167 秒 |
| Milvus collection | `document_chunks_v3_2560` |
| MySQL Chunk | 1246，全部 `embedded` |
| Milvus Chunk | 1246 |
| 两库 Chunk ID 集合 | 完全一致 |
| 是否重跑 MinerU/VLM | 否 |

`index-chunks` 会先运行 Chunk 硬门控，再完成全部向量请求；向量成功后才更新存储。MySQL 使用单事务只替换该文档的 Chunk，不删除 pages、blocks、assets、revisions 和 workflow events。Milvus 使用新的 `document_chunks_v3_2560` collection，不覆盖旧版 collection。

需要诚实说明：MySQL 和 Milvus 之间没有分布式事务。当前实际结果已经通过数量和 ID 集合复核为一致；生产环境若要求故障原子性，应增加索引版本表、staging collection 和 active-version 指针切换。

## 4. 独立问题集

subagent 仅根据原始 PDF 构造了 48 题：

| 类型 | 数量 |
| --- | ---: |
| 正文 | 14 |
| 表格 | 19 |
| 图表 | 7 |
| 脚注 | 5 |
| 跨页 | 3 |

难度分布为 easy 2、medium 15、hard 31。题目包含预期答案、PDF 物理页码以及必须召回的关键词，覆盖第 44、84、100—107、115 页等复杂位置。

## 5. 指标定义

- `page_hit`：TopK 中至少一个结果页与标注物理页相交。
- `keyword_coverage`：TopK 所有证据文本的并集覆盖了多少必需关键词。
- `content_success`：关键词覆盖率至少 60%。
- `strict_keyword_success`：全部必需关键词均被 TopK 证据覆盖。
- `grounded_success`：内容成功，且目标页没有失配。

数字匹配采用边界检查，`134.9` 不会错误命中 `134.91`。表格允许“单元格中的纯数字 + 同一 Chunk 的表级单位”联合满足 `440513.31亿元`，但不会把别的数字当作命中。

## 6. 优化前后召回结果

优化前基线是保存的纯向量召回结果，并使用最终相同的数字边界评分器离线重算，所以**评分口径一致**。但历史结果中有 3 个已被后续阅读顺序修复替换的旧 Chunk ID，因此它只能作为系统级历史基线，不能当成“同一索引只开关混合检索”的严格消融实验，也不能把全部提升归因于 RRF。

| 版本 | K | 页命中 | 内容成功 | 联合成功 | 严格关键词 | 平均覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯向量基线 | 1 | 93.75% | 89.58% | 89.58% | 81.25% | 88.78% |
| 最终混合检索 | 1 | 97.92% | 91.67% | 91.67% | 81.25% | 91.33% |
| 纯向量基线 | 3 | 93.75% | 93.75% | 93.75% | 89.58% | 93.58% |
| 最终混合检索 | 3 | **100%** | **100%** | **100%** | **100%** | **100%** |
| 纯向量基线 | 5 | 93.75% | 95.83% | 93.75% | 89.58% | 94.97% |
| 最终混合检索 | 5 | **100%** | **100%** | **100%** | **100%** | **100%** |

最终检索流程为：

1. 向量召回 Top60；
2. 对单文档 1246 个 Chunk 做字符 bigram 词法打分，取 Top30；
3. 用 RRF 合并两路候选，保留语义近似和精确数字/专名两类证据；
4. 对候选并集调用 reranker；
5. 以 reranker 分数为主，dense/lexical rank 各只增加最多 `0.01/sqrt(rank)` 的稳定项；
6. 同一表格父节点最多保留 3 条 row/group，支持“总资产和总负债分别是多少”这种多行问题。

第 5 步是一次回归测试后的修正。曾尝试将 RRF 再作为最终主排序，虽然修复了一道双行表格题，却使跨页证据退到 Top5/Top10，因此已撤销。最终方案只允许两路原始排名对 reranker 的近似并列结果做最多 0.02 的小幅稳定，不让弱词法候选越过明显相关的正文。

## 7. Top1 为什么不是 100%

最终 Top1 有 4 个 `grounded_success=false`，但全部在 Top3 得到完整证据：

1. `Q001`：问题标注第 4 页的 `134.9万亿元`，Top1 命中第 14 页更精确的 `134.91万亿元`。数字边界规则正确地没有把二者判为相同，Top3 返回第 4 页原文。
2. `Q003`：美国、欧元区、日本三组图表事实分散在多个 `chart_fact`，一个 Chunk 无法覆盖全部国家数值。
3. `Q021`：票据与黄金市场措施横跨第 60—61 页，Top1 只能覆盖一页，Top3 合并后完整。
4. `Q033`：压力测试图中的三个情景值拆为多个图表事实，Top1 只覆盖部分，Top3 完整。

这说明默认 RAG 建议取 `top_k>=3`，并允许回答器合并同一父图表、同一表格或相邻页的证据，同时保留页码引用。

## 8. 为什么是 1200 字符和 120 overlap

这是**工程基线，不是本次实验得到的最优值**。

- `1200` 按 Python `len(str)` 统计 Unicode code point，不是 token，也不是字节。它用于让一个正文 Chunk 通常容纳若干完整中文句，同时避免整页或整节进入一个向量。
- `120` 是 1200 的 10%，用作句子窗口的边界缓冲。普通窗口会优先回取完整句子，因此实际 overlap 通常是“至少 120 字符的完整尾部句子”，不一定恰好 120；只有无法按标点拆开的超长字符串才使用 1200 窗口、1080 步长进行精确硬切。
- 表格、图表、脚注不统一套用 1200/120。表格按 summary、row group、row 多粒度构建；图表按 summary、fact 构建；脚注单独成块。

本 PDF 的 329 个 `text_segment` 中，显示文本中位数为 175、P95 为 489、最大值为 1012。多数内容在碰到 1200 上限前已经因章节、页面、栏或 block 状态边界结束。因此本次 48 题可以证明当前**完整 Chunk 与检索链路**在该文档上可用，不能证明 1200/120 是通用最优参数。

若面试官追问“依据是什么”，准确回答是：

> 1200/120 是中文长文的可配置工程起点，120 是 10% 边界上下文；我没有把它包装成实验最优值。当前样本中正文 P95 只有 489 字符，说明语义边界比长度上限更常触发。若要得出最优值，需要固定解析结果、模型与检索策略，对 600/60、900/90、1200/120、1600/160 等隔离索引做同题 ablation，同时比较 Recall@K、MRR/nDCG、向量数、延迟和重复上下文。

## 9. 评测边界

本次结果有四个边界，不能省略：

- 这是单份 PDF、48 个问题的定向回归集，不是跨文档统计结论。
- 关键词与目标页命中证明“证据被取回”，不证明后续大模型一定生成正确答案。
- 它不等于 OCR 字符准确率或表格 cell 准确率；要评估解析准确率仍需字符级、block 级和 cell 级人工 ground truth。
- 126 个 provisional Chunk 仍携带 `requires_human_review`/`retrieval_tier`，高风险问答应展示提示或只允许 verified 证据进入自动结论。

## 10. 复现命令

```bash
cd /path/to/document-evidence-rag

PYTHONPATH=src python \
  -m mineru_vlm_rag.cli audit-chunks \
  outputs/4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393/chunks_v3_review/all_chunks.json

PYTHONPATH=src python \
  -m mineru_vlm_rag.cli index-chunks \
  4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393 \
  --input outputs/4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393/chunks_v3_review/all_chunks.json

PYTHONPATH=src python \
  -m mineru_vlm_rag.cli --verbose eval-retrieval \
  4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393 \
  --questions evaluation/pdf_ground_truth_questions.json \
  --output-dir outputs/4e49437bfd1aa467396ef4f92afac40c20bfbe19a47c9a49ab177ad3d51e0393/chunks_v3_review/evaluation \
  --limit 10
```

`index-chunks` 会真实调用 Embedding 并覆盖该文档在 v3 collection 与 MySQL 中的 Chunk；只想查看现有结果时不要重复执行该命令。
