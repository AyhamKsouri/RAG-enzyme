import numpy as np 
from typing import List 
from rank_bm25 import BM25Okapi 

from src.domain.interfaces import VectorStorePort 
from src.domain.models import Fragment, SearchResult 


class HybridVectorStore(VectorStorePort): 
    """ 
    Hybrid search combining: 
    - Dense semantic search (cosine similarity on embeddings) 
    - Sparse keyword search (BM25 on tokenized text) 

    Final score = α * semantic_score + (1-α) * bm25_score 

    Why hybrid: 
    - Semantic alone misses exact product names ("GOX 110") 
    - BM25 alone misses conceptual queries ("enzymes for dough strength") 
    - Together: best of both worlds 
    """ 

    def __init__(self, semantic_weight: float = 0.6): 
        """ 
        Args: 
            semantic_weight: Weight for semantic score (0-1). 
                             1-semantic_weight goes to BM25. 
                             0.6 = slight semantic preference, strong keyword boost. 
         """ 
        self._fragments: List[Fragment] = [] 
        self._embedding_matrix: np.ndarray | None = None 
        self._bm25: BM25Okapi | None = None 
        self._semantic_weight = semantic_weight 
        self._bm25_weight = 1.0 - semantic_weight 

    def index_fragments(self, fragments: List[Fragment]) -> None: 
        if not fragments: 
            raise ValueError("Cannot index an empty fragment list.") 

        missing = [f.fragment_id for f in fragments if f.embedding is None] 
        if missing: 
            raise ValueError(f"Fragments missing embeddings: {missing}") 

        self._fragments = fragments 
        self._embedding_matrix = np.stack([f.embedding for f in fragments]) 

        # Build BM25 index on lowercased word tokens 
        tokenized_corpus = [ 
            fragment.text.lower().split() 
            for fragment in fragments 
        ] 
        self._bm25 = BM25Okapi(tokenized_corpus) 

        print(f"[HybridStore] Indexed {len(fragments)} fragments — " 
              f"semantic weight: {self._semantic_weight}, " 
              f"BM25 weight: {self._bm25_weight}") 

    def is_ready(self) -> bool: 
        """In-memory hybrid store is ready only when explicitly indexed this session.""" 
        return self._bm25 is not None 

    def get_indexed_file_hashes(self) -> dict[str, str]:
        """In-memory store does not persist hashes between sessions."""
        return {}

    def search( 
        self, 
        query_text: str,            # Used for BM25 
        query_embedding: np.ndarray, # Used for semantic 
        top_k: int = 3, 
    ) -> List[SearchResult]: 
        """ 
        Execute hybrid search combining semantic + BM25 scores. 

        Args: 
            query_text: Raw query text (for BM25 tokenization). 
            query_embedding: Normalized embedding of the query. 
            top_k: Number of results to return. 

        Returns: 
            Top-k results ranked by combined score, descending. 
        """ 
        if self._embedding_matrix is None or self._bm25 is None: 
            raise RuntimeError("Store is empty. Call index_fragments() first.") 

        # ── Semantic scores (cosine via dot product on normalized vectors) ── 
        semantic_scores = self._embedding_matrix @ query_embedding  # Shape: (N,) 

        # Normalize semantic scores to [0, 1] 
        sem_min, sem_max = semantic_scores.min(), semantic_scores.max() 
        sem_range = sem_max - sem_min if sem_max > sem_min else 1.0 
        semantic_scores_norm = (semantic_scores - sem_min) / sem_range 

        # ── BM25 keyword scores ─────────────────────────────────────────── 
        query_tokens = query_text.lower().split() 
        bm25_scores = np.array(self._bm25.get_scores(query_tokens))  # Shape: (N,) 

        # Normalize BM25 scores to [0, 1] 
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max() 
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0 
        bm25_scores_norm = (bm25_scores - bm25_min) / bm25_range 

        # ── Combined score ──────────────────────────────────────────────── 
        combined_scores = ( 
            self._semantic_weight * semantic_scores_norm 
            + self._bm25_weight * bm25_scores_norm 
        ) 

        # Get top-k by combined score 
        top_k_indices = np.argpartition(combined_scores, -top_k)[-top_k:] 
        top_k_sorted = top_k_indices[np.argsort(combined_scores[top_k_indices])[::-1]] 

        return [ 
            SearchResult( 
                fragment=self._fragments[idx], 
                similarity_score=float(combined_scores[idx]), 
            ) 
            for idx in top_k_sorted 
        ] 
