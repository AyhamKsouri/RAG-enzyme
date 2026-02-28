# src/infrastructure/vector_store.py

import numpy as np
from typing import List

from src.domain.interfaces import VectorStorePort
from src.domain.models import Fragment, SearchResult


class InMemoryVectorStore(VectorStorePort):
    """
    In-memory vector store with two search modes:
    - Standard cosine similarity (dot product on normalized vectors)
    - MMR (Maximal Marginal Relevance): balances relevance + source diversity
    """

    def __init__(self, use_mmr: bool = True, mmr_lambda: float = 0.7):
        self._fragments: List[Fragment] = []
        self._embedding_matrix: np.ndarray | None = None
        self._use_mmr = use_mmr
        # lambda=0.7 → 70% relevance, 30% diversity. Tune between 0.5–0.9.
        self._mmr_lambda = mmr_lambda

    def index_fragments(self, fragments: List[Fragment]) -> None:
        if not fragments:
            raise ValueError("Cannot index an empty fragment list.")

        missing = [f.fragment_id for f in fragments if f.embedding is None]
        if missing:
            raise ValueError(f"Fragments missing embeddings: {missing}")

        self._fragments = fragments
        self._embedding_matrix = np.stack([f.embedding for f in fragments])
        print(f"[VectorStore] Indexed {len(fragments)} fragments. "
              f"Matrix shape: {self._embedding_matrix.shape}")

    def is_ready(self) -> bool: 
        """In-memory store is ready only when explicitly indexed this session.""" 
        return self._embedding_matrix is not None 

    def get_indexed_file_hashes(self) -> dict[str, str]:
        """In-memory store does not persist hashes between sessions."""
        return {}

    def get_document_stats(self) -> List[dict]:
        """In-memory store aggregates current fragments."""
        counts = {}
        for f in self._fragments:
            counts[f.source_document] = counts.get(f.source_document, 0) + 1
        return [{"filename": name, "count": count} for name, count in sorted(counts.items())]

    def search( 
        self, 
        query_text: str,           # Accepted but unused — pure semantic store 
        query_embedding: np.ndarray, 
        top_k: int = 3, 
    ) -> List[SearchResult]: 
        # query_text intentionally ignored — this store uses embeddings only 
        if self._embedding_matrix is None: 
            raise RuntimeError("Vector store is empty. Call index_fragments() first.") 
 
        if self._use_mmr:
            return self._search_mmr(query_embedding, top_k)
        return self._search_cosine(query_embedding, top_k)

    # ─── Search Strategies ────────────────────────────────────────────────────

    def _search_cosine(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Pure cosine similarity — fast but may return duplicates from same source."""
        scores = self._embedding_matrix @ query_embedding
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices_sorted = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(fragment=self._fragments[i], similarity_score=float(scores[i]))
            for i in top_indices_sorted
        ]

    def _search_mmr(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]: 
        """ 
        Maximal Marginal Relevance search. 
 
        Selects results balancing relevance to query and diversity from 
        already-selected fragments. 
 
        Formula: MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected)) 
 
        Key fix: scores are normalized to [0,1] range before MMR computation 
        so that relevance and redundancy penalties are on the same scale. 
        """ 
        query_scores = self._embedding_matrix @ query_embedding  # Shape: (N,) 
 
        # Work with full candidate pool for small corpora (<500 docs) 
        # For larger corpora, pre-filter to top-100 for performance 
        pool_size = min(100, len(self._fragments)) 
        candidate_indices = list( 
            np.argpartition(query_scores, -pool_size)[-pool_size:] 
        ) 
 
        # Normalize scores to [0, 1] so λ is meaningful across different corpora 
        score_min = query_scores[candidate_indices].min() 
        score_max = query_scores[candidate_indices].max() 
        # For small pools or single documents, ensure we don't divide by zero
        score_range = score_max - score_min if score_max > score_min else 1.0 
 
        def normalized_relevance(idx: int) -> float: 
            # If all scores are the same, they are all equally relevant.
            if score_max == score_min:
                return 1.0
            return float((query_scores[idx] - score_min) / score_range) 
 
        selected_indices: List[int] = [] 
        selected_embeddings: List[np.ndarray] = [] 
 
        for _ in range(top_k): 
            best_index = None 
            best_mmr_score = -np.inf 
 
            for idx in candidate_indices: 
                if idx in selected_indices: 
                    continue 
 
                relevance = normalized_relevance(idx) 
 
                if selected_embeddings: 
                    selected_matrix = np.stack(selected_embeddings) 
                    # Max similarity to any already-selected fragment 
                    similarities = selected_matrix @ self._fragments[idx].embedding
                    redundancy = float(np.max(similarities))
                    # Clamp redundancy to [0, 1] for cosine similarity
                    redundancy = max(0.0, min(1.0, redundancy))
                else: 
                    redundancy = 0.0 
 
                mmr_score = ( 
                    self._mmr_lambda * relevance 
                    - (1 - self._mmr_lambda) * redundancy 
                ) 
 
                # Source diversity boost: penalize results from already-selected sources
                selected_sources = {self._fragments[i].source_document for i in selected_indices}
                if self._fragments[idx].source_document in selected_sources:
                    # Apply a small penalty to discourage picking from the same document
                    mmr_score -= 0.1

                if mmr_score > best_mmr_score: 
                    best_mmr_score = mmr_score 
                    best_index = idx 
 
            if best_index is None: 
                break 
 
            selected_indices.append(best_index) 
            selected_embeddings.append(self._fragments[best_index].embedding) 
 
        return [ 
            SearchResult( 
                fragment=self._fragments[i], 
                similarity_score=float(query_scores[i]), 
            ) 
            for i in selected_indices 
        ]