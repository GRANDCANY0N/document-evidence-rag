from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from mineru_vlm_rag.adapters.siliconflow import SiliconFlowClient
from mineru_vlm_rag.persistence import MilvusRepository
from mineru_vlm_rag.settings import Settings


class QueryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.milvus = MilvusRepository(
            settings.milvus_uri,
            collection_prefix=str(settings.pipeline.get("milvus_collection_prefix", "document_chunks_v3")),
        )
        self.silicon = SiliconFlowClient(
            api_key=settings.siliconflow_api_key,
            base_url=settings.vlm["api_base"],
            vlm_model=settings.vlm_model,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
        )
        self._lexical_cache: dict[tuple[int, str | None, str | None], list[dict[str, Any]]] = {}

    @staticmethod
    def _lexical_terms(value: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", value or "").lower().replace("\\%", "%")
        terms: list[str] = []
        for token in re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]+", normalized):
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                if len(token) == 1:
                    terms.append(token)
                else:
                    terms.extend(token[index:index + 2] for index in range(len(token) - 1))
            else:
                terms.append(token)
        return terms

    def _lexical_candidates(
        self,
        query: str,
        *,
        dimension: int,
        document_id: str | None,
        block_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        key = (dimension, document_id, block_type)
        corpus = self._lexical_cache.get(key)
        if corpus is None:
            corpus = self.milvus.list_chunks(
                dimension,
                document_id=document_id,
                block_type=block_type,
            )
            self._lexical_cache[key] = corpus
        query_counts = Counter(self._lexical_terms(query))
        if not query_counts or not corpus:
            return []
        document_terms = [Counter(self._lexical_terms(str(item.get("text") or ""))) for item in corpus]
        document_frequency = Counter(
            term
            for counts in document_terms
            for term in query_counts
            if term in counts
        )
        total = len(corpus)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item, counts in zip(corpus, document_terms, strict=True):
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1.0 + (total + 0.5) / (document_frequency[term] + 0.5))
                score += query_frequency * inverse_frequency * (1.0 + math.log(frequency))
            if score:
                # Mild length normalization prevents table summaries from
                # winning only because they repeat every row label.
                score /= math.sqrt(max(1.0, len(counts)))
                value = dict(item)
                value["lexical_score"] = score
                ranked.append((score, value))
        ranked.sort(key=lambda pair: (-pair[0], int(pair[1].get("document_order") or 0)))
        return [item for _, item in ranked[:limit]]

    @staticmethod
    def _final_rank_score(
        *,
        relevance_score: float,
        dense_rank: int | None,
        lexical_rank: int | None,
    ) -> float:
        """Stabilize near-tied reranker scores with bounded rank agreement.

        RRF is already used to construct the candidate union.  Applying it a
        second time as the dominant final score can move weak lexical matches
        above genuinely relevant cross-page evidence.  The API reranker is
        therefore kept as the primary signal; dense and lexical ranks can add
        at most 0.02 in total and only reorder near ties.
        """

        score = float(relevance_score)
        if dense_rank is not None:
            score += 0.01 / math.sqrt(max(1, dense_rank))
        if lexical_rank is not None:
            score += 0.01 / math.sqrt(max(1, lexical_rank))
        return score

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        block_type: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        vector = self.silicon.embed([query], dimensions=int(self.settings.embedding["dimensions"]))[0]
        candidate_pool = max(limit * 6, 60)
        dense_candidates = self.milvus.search(
            vector,
            limit=candidate_pool,
            document_id=document_id,
            block_type=block_type,
        )
        lexical_candidates = self._lexical_candidates(
            query,
            dimension=len(vector),
            document_id=document_id,
            block_type=block_type,
            limit=max(limit * 3, 30),
        )
        combined: dict[str, dict[str, Any]] = {}
        scores: Counter[str] = Counter()
        for rank, candidate in enumerate(dense_candidates, start=1):
            candidate = dict(candidate)
            candidate["dense_rank"] = rank
            chunk_id = str(candidate.get("id") or "")
            combined[chunk_id] = candidate
            scores[chunk_id] += 1.0 / (60 + rank)
        for rank, candidate in enumerate(lexical_candidates, start=1):
            chunk_id = str(candidate.get("id") or "")
            value = combined.setdefault(chunk_id, dict(candidate))
            value["lexical_rank"] = rank
            value["lexical_score"] = candidate.get("lexical_score")
            scores[chunk_id] += 1.0 / (60 + rank)
        candidates = sorted(
            combined.values(),
            key=lambda candidate: (
                -scores[str(candidate.get("id") or "")],
                int(candidate.get("document_order") or 0),
            ),
        )
        if not candidates:
            return []
        # Ask the reranker for enough candidates before parent-level dedupe;
        # otherwise several exact table rows from one logical table can leave
        # the caller with fewer than ``limit`` distinct evidence groups.
        rerank_top_n = len(candidates)
        reranked = self.silicon.rerank(
            query,
            [item["text"] for item in candidates],
            top_n=rerank_top_n,
        )
        fused: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        for rerank_position, item in enumerate(reranked, start=1):
            candidate = dict(candidates[int(item["index"])])
            fused.append(
                (
                    self._final_rank_score(
                        relevance_score=float(item.get("relevance_score") or 0.0),
                        dense_rank=candidate.get("dense_rank"),
                        lexical_rank=candidate.get("lexical_rank"),
                    ),
                    rerank_position,
                    candidate,
                    item,
                )
            )
        fused.sort(
            key=lambda value: (
                -value[0],
                value[1],
                int(value[2].get("document_order") or 0),
            )
        )
        results: list[dict[str, Any]] = []
        parent_counts: Counter[str] = Counter()
        for fusion_score, rerank_position, candidate, item in fused:
            parent = candidate.get("parent_id") or candidate.get("id")
            # A question can ask for two or more rows from the same table
            # (for example total assets and total liabilities).  Keep a small
            # per-parent cap instead of collapsing the whole table to one row.
            parent_cap = 3 if candidate.get("chunk_role") in {"table_row", "table_row_group"} else 2
            if parent_counts[str(parent)] >= parent_cap:
                continue
            parent_counts[str(parent)] += 1
            candidate["rank"] = len(results) + 1
            candidate["rerank_position"] = rerank_position
            candidate["final_rank_score"] = fusion_score
            candidate["retrieval_sources"] = [
                source for source, present in (
                    ("dense", candidate.get("dense_rank") is not None),
                    ("lexical", candidate.get("lexical_rank") is not None),
                ) if present
            ]
            candidate["relevance_score"] = item.get("relevance_score")
            results.append(candidate)
            if len(results) >= limit:
                break
        return results

    def close(self) -> None:
        self.silicon.close()
        self.milvus.close()
