"""Hybrid search combining BM25 and vector search using Reciprocal Rank Fusion."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .bm25_indexer import BM25Indexer
from .vector_indexer import VectorIndexer
from ..utils.config import get_config


@dataclass
class SearchResult:
    """A search result with combined ranking."""
    id: int
    content: str
    path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0


class HybridSearch:
    """Hybrid search combining BM25 and vector search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        db_path: Path,
        vector_path: Path,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        """
        Initialize hybrid search.

        Args:
            db_path: Path to SQLite database
            vector_path: Path to vector index
            bm25_weight: Weight for BM25 scores in RRF
            vector_weight: Weight for vector scores in RRF
        """
        self.bm25_indexer = BM25Indexer(db_path)
        self.vector_indexer = VectorIndexer(vector_path)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(
        self,
        query: str,
        limit: int = 10,
        path_filter: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Search query
            limit: Maximum number of results
            path_filter: Optional path filter
            rerank: Override config setting for re-ranking (None uses config)

        Returns:
            List of SearchResult objects ranked by combined RRF score,
            or re-ranked by cross-encoder if enabled.
        """
        config = get_config()
        bm25_weight = self.bm25_weight or config.bm25_weight
        vector_weight = self.vector_weight or config.vector_weight

        # Determine if re-ranking is enabled
        use_rerank = config.rerank_enabled if rerank is None else rerank

        # Determine number of candidates to retrieve
        if use_rerank:
            candidates_count = config.rerank_candidates
        else:
            candidates_count = min(limit * 2, 50)

        # Get BM25 results
        bm25_results = self.bm25_indexer.search(
            query,
            limit=candidates_count,
            path_filter=path_filter,
        )

        # Get vector results
        vector_results = self.vector_indexer.search(
            query,
            k=candidates_count,
        )

        # Combine using Reciprocal Rank Fusion
        results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
        )

        # Stage 2: Sort and reduce to rrf_limit before re-ranking
        results.sort(key=lambda x: x.combined_score, reverse=True)
        if use_rerank:
            results = results[:config.rrf_limit]

        # Stage 3: Apply re-ranking if enabled
        if use_rerank and results:
            results = self._rerank_results(query, results, limit, config)
        else:
            # No re-ranking: limit results
            results = results[:limit]

        # Add rank
        for i, result in enumerate(results, 1):
            result.rank = i

        return results

    def _rerank_results(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
        config,
    ) -> List[SearchResult]:
        """Apply cross-encoder re-ranking to candidates."""
        try:
            from ..embedding.cross_encoder import get_cross_encoder

            cross_encoder = get_cross_encoder(config.rerank_model)
            results = cross_encoder.rerank(
                query,
                candidates,
                top_k=top_k,
                threshold=config.rerank_threshold,
            )
            return results
        except Exception as e:
            # Fall back to RRF-only results if re-ranking fails
            import sys
            print(f"Warning: Re-ranking failed, falling back to RRF: {e}", file=sys.stderr)
            candidates.sort(key=lambda x: x.combined_score, reverse=True)
            return candidates[:top_k]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[dict],
        vector_results: List[Tuple[int, float]],
        bm25_weight: float,
        vector_weight: float,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each ranking

        This method is robust when one retriever returns no results - it simply
        uses the scores from the other retriever.

        Args:
            bm25_results: BM25 search results
            vector_results: Vector search results (chunk_id, distance)
            bm25_weight: Weight for BM25 ranks
            vector_weight: Weight for vector ranks
            k: RRF constant (default 60)

        Returns:
            Combined and deduplicated results
        """
        # Map chunk_id to result
        results_map: Dict[int, SearchResult] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result["id"]
            if chunk_id not in results_map:
                results_map[chunk_id] = SearchResult(
                    id=chunk_id,
                    content=result["content"],
                    path=result["path"],
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    language=result["language"],
                    chunk_type=result["chunk_type"],
                )
            results_map[chunk_id].bm25_score = bm25_weight / (k + rank)

        # Process vector results
        for rank, (chunk_id, distance) in enumerate(vector_results, 1):
            if chunk_id not in results_map:
                # Need to fetch chunk info from BM25 indexer
                chunk_info = self.bm25_indexer.db.get_chunk_by_id(chunk_id)
                if chunk_info:
                    results_map[chunk_id] = SearchResult(
                        id=chunk_id,
                        content=chunk_info["content"],
                        path=chunk_info["path"],
                        start_line=chunk_info["start_line"],
                        end_line=chunk_info["end_line"],
                        language=chunk_info["language"],
                        chunk_type=chunk_info["chunk_type"],
                    )
                else:
                    continue
            results_map[chunk_id].vector_score = vector_weight / (k + rank)

        # Calculate combined scores
        for result in results_map.values():
            result.combined_score = result.bm25_score + result.vector_score

        return list(results_map.values())

    def close(self) -> None:
        """Close all connections."""
        self.bm25_indexer.close()