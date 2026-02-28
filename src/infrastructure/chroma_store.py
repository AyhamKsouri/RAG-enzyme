# src/infrastructure/chroma_store.py

import json
import numpy as np
from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.domain.interfaces import VectorStorePort
from src.domain.models import Fragment, SearchResult


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_FINGERPRINT_KEY  = "embedding_model_name"
FILE_HASHES_KEY        = "indexed_file_hashes"
COLLECTION_NAME        = "document_fragments"
METADATA_SENTINEL_ID   = "__metadata__"

# Retrieve this many candidates from ChromaDB before MMR + BM25 reranking.
# Larger pool = better diversity at the cost of slightly more reranking work.
MMR_CANDIDATE_MULTIPLIER = 5


class ChromaVectorStore(VectorStorePort):
    """
    Production-grade persistent vector store combining:

    ┌─────────────────────────────────────────────────────┐
    │  ChromaDB (disk)  →  cosine similarity candidates   │
    │  BM25 (memory)    →  keyword relevance reranking    │
    │  MMR  (memory)    →  diversity selection            │
    └─────────────────────────────────────────────────────┘

    Search pipeline:
        1. ChromaDB fetches top-(k * MMR_CANDIDATE_MULTIPLIER) candidates
           via HNSW cosine similarity
        2. BM25 re-scores each candidate — boosts exact keyword matches
           (e.g. product names like "GOX 110")
        3. MMR selects final top-k from reranked pool, penalizing
           near-duplicate fragments

    Persistence features:
        - Model fingerprinting: detects embedding model changes → auto reindex
        - File hash tracking: detects new/modified source files → incremental index
        - Idempotent upserts: safe to call index_fragments() multiple times
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name: str,
        semantic_weight: float = 0.6,
        mmr_lambda: float = 0.6,
    ):
        """
        Args:
            persist_directory:    Path for ChromaDB on-disk storage.
            embedding_model_name: Current embedding model identifier.
                                  Used to detect stale stored embeddings.
            semantic_weight:      Weight for semantic score in hybrid formula.
                                  (1 - semantic_weight) goes to BM25.
            mmr_lambda:           MMR relevance vs diversity balance.
                                  1.0 = pure relevance, 0.0 = pure diversity.
        """
        self._persist_directory    = persist_directory
        self._embedding_model_name = embedding_model_name
        self._semantic_weight      = semantic_weight
        self._mmr_lambda           = mmr_lambda

        # BM25 index lives in memory — rebuilt on each index_fragments() call.
        # Lightweight: 93 fragments = negligible memory footprint.
        self._bm25:        Optional[BM25Okapi] = None
        self._bm25_ids:    List[str]           = []
        self._bm25_texts:  List[str]           = []

        # Ensure the directory exists.
        path = Path(persist_directory)
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Failed to initialize ChromaDB: path '{persist_directory}' is a file.")
        path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at '{persist_directory}'.\n"
                f"The database may be locked by another process or corrupted.\n"
                f"Fix: close other running instances, or delete '{persist_directory}'.\n"
                f"Original error: {error}"
            ) from error

        fragment_count = self._real_fragment_count()
        print(
            f"[ChromaStore] Connected to '{persist_directory}'. "
            f"Collection has {fragment_count} fragments."
        )

        # Rebuild BM25 from persisted fragments so keyword search works
        # immediately on subsequent runs without re-encoding
        if fragment_count > 0:
            self._rebuild_bm25_from_store()

    # ─── VectorStorePort: Core Interface ─────────────────────────────────────

    def is_ready(self) -> bool:
        """
        True only when:
        1. Store has indexed fragments (excluding the sentinel document), AND
        2. Stored model fingerprint matches the current model.
        """
        if self._real_fragment_count() == 0:
            return False

        stored_model = self._get_metadata_value(MODEL_FINGERPRINT_KEY)
        if stored_model != self._embedding_model_name:
            print(
                f"[ChromaStore] ⚠ Model mismatch detected!\n"
                f"  Stored : '{stored_model}'\n"
                f"  Current: '{self._embedding_model_name}'\n"
                f"  → Full reindex required."
            )
            return False

        return True

    def get_indexed_file_hashes(self) -> dict[str, str]:
        """
        Return { filename: sha256_hash } map saved during the last index build.
        Empty dict if no metadata has been stored yet.
        """
        raw = self._get_metadata_value(FILE_HASHES_KEY)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def get_document_stats(self) -> List[dict]:
        """
        Fetch all metadatas from the store and aggregate counts by document.
        Excludes the sentinel document.
        """
        results = self._collection.get(
            include = ["metadatas"],
            where   = {"source_document": {"$ne": METADATA_SENTINEL_ID}},
        )

        metadatas = results["metadatas"]
        counts = {}
        for meta in metadatas:
            doc = meta.get("source_document", "unknown")
            counts[doc] = counts.get(doc, 0) + 1

        return [
            {"filename": name, "count": count} 
            for name, count in sorted(counts.items())
        ]

    def index_fragments(self, fragments: List[Fragment]) -> None:
        """
        Upsert fragments into ChromaDB and rebuild the in-memory BM25 index.

        Safe to call multiple times — upsert semantics prevent duplication.
        Automatically clears stale data if the embedding model has changed.
        """
        if not fragments:
            raise ValueError("Cannot index an empty fragment list.")

        missing = [f.fragment_id for f in fragments if f.embedding is None]
        if missing:
            raise ValueError(f"Fragments missing embeddings: {missing[:5]}")

        # Wipe collection if stored model doesn't match — embeddings are stale
        if self._real_fragment_count() > 0:
            stored_model = self._get_metadata_value(MODEL_FINGERPRINT_KEY)
            if stored_model and stored_model != self._embedding_model_name:
                print("[ChromaStore] Clearing stale collection before reindex...")
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

        print(f"[ChromaStore] Upserting {len(fragments)} fragments...")

        ids        = [f.fragment_id       for f in fragments]
        embeddings = [f.embedding.tolist() for f in fragments]
        documents  = [f.text              for f in fragments]
        metadatas  = [{
            "source_document": f.source_document,
            "page_number": f.page_number
        } for f in fragments]

        # Batch upsert — Python slicing handles non-divisible sizes gracefully
        batch_size = 500
        for start in range(0, len(fragments), batch_size):
            self._collection.upsert(
                ids        = ids       [start : start + batch_size],
                embeddings = embeddings[start : start + batch_size],
                documents  = documents [start : start + batch_size],
                metadatas  = metadatas [start : start + batch_size],
            )

        # Rebuild BM25 from the full updated corpus
        self._rebuild_bm25_from_store()

        # Automatically save model fingerprint after successful indexing
        # This ensures is_ready() returns True for the current session
        self.save_index_metadata(self.get_indexed_file_hashes())

        print(
            f"[ChromaStore] ✓ Upserted {len(fragments)} fragments. "
            f"Total in store: {self._real_fragment_count()}"
        )

    def save_index_metadata(self, file_hashes: dict[str, str]) -> None:
        """
        Persist model fingerprint and file hashes as a sentinel document.
        Called by main.py after a successful index build.
        """
        # Fetch a sample to get the correct embedding dimension
        sample = self._collection.get(limit=1, include=["embeddings"])
        dim = len(sample["embeddings"][0]) if (sample["embeddings"] is not None and len(sample["embeddings"]) > 0) else 384

        self._collection.upsert(
            ids        = [METADATA_SENTINEL_ID],
            embeddings = [[0.0] * dim],
            documents  = [METADATA_SENTINEL_ID],
            metadatas  = [{
                MODEL_FINGERPRINT_KEY: self._embedding_model_name,
                FILE_HASHES_KEY:       json.dumps(file_hashes),
            }],
        )

    def delete_document(self, filename: str) -> None:
        """
        Remove all fragments associated with a document and update metadata.
        """
        # 1. Remove from ChromaDB
        self._collection.delete(where={"source_document": filename})
        
        # 2. Update indexed file hashes metadata
        current_hashes = self.get_indexed_file_hashes()
        if filename in current_hashes:
            del current_hashes[filename]
            self.save_index_metadata(current_hashes)
            
        # 3. Rebuild BM25 index
        self._rebuild_bm25_from_store()
        
        print(f"[ChromaStore] Deleted document '{filename}' and updated metadata.")

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        True Parallel Hybrid Search pipeline:
            1. Fetch top-K candidates from ChromaDB (Dense)
            2. Fetch top-K candidates from BM25 (Sparse keyword)
            3. Combine candiate pools to ensure exact matches aren't lost
            4. Rescore combined pool with hybrid formula
            5. MMR diversity selection
        """
        if self._real_fragment_count() == 0:
            raise RuntimeError("ChromaStore is empty. Call index_fragments() first.")

        candidate_count = min(
            top_k * MMR_CANDIDATE_MULTIPLIER,
            self._real_fragment_count(),
        )

        # ── Step 1: Fetch Dense candidates from ChromaDB ──────────────────────
        raw_results = self._collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = candidate_count,
            include          = ["documents", "metadatas", "distances", "embeddings"],
            where            = {"source_document": {"$ne": METADATA_SENTINEL_ID}},
        )
        dense_candidates = {
            c["result"].fragment.fragment_id: c 
            for c in self._map_to_candidates(raw_results)
        }

        # ── Step 2: Fetch Sparse candidates from BM25 ─────────────────────────
        sparse_candidates = {}
        if self._bm25 is not None and len(self._bm25_ids) > 0:
            query_tokens = self._tokenize(query_text)
            bm25_scores = self._bm25.get_scores(query_tokens)
            
            # Get top indices for BM25
            top_sparse_n = min(candidate_count, len(bm25_scores))
            top_indices = np.argsort(bm25_scores)[-top_sparse_n:][::-1]
            
            # We need the embeddings for MMR, so we bulk-fetch the missing ones from Chroma
            sparse_ids_to_fetch = [
                self._bm25_ids[i] for i in top_indices 
                if bm25_scores[i] > 0 and self._bm25_ids[i] not in dense_candidates
            ]
            
            if sparse_ids_to_fetch:
                missing_results = self._collection.get(
                    ids=sparse_ids_to_fetch,
                    include=["documents", "metadatas", "embeddings"]
                )
                
                # Reconstruct candidate objects for the sparse results
                for fid, text, metadata, embedding in zip(
                    missing_results["ids"],
                    missing_results["documents"],
                    missing_results["metadatas"],
                    missing_results["embeddings"],
                ):
                    sparse_candidates[fid] = {
                        "result": SearchResult(
                            fragment=Fragment(
                                fragment_id=fid,
                                source_document=metadata.get("source_document", "unknown"),
                                text=text,
                                page_number=metadata.get("page_number", 1),
                            ),
                            similarity_score=0.0, # Will be computed below
                        ),
                        "embedding": np.array(embedding, dtype=np.float32),
                    }

        # ── Step 3: Combine and Rerank ────────────────────────────────────────
        # Merge dictionaries (sparse_candidates adds any missing exact matches)
        combined_candidates = list({**dense_candidates, **sparse_candidates}.values())
        
        if not combined_candidates:
            return []

        # Calculate absolute cosine similarities for any records missing them
        for c in combined_candidates:
            if c["result"].similarity_score == 0.0:
                # Cosine similarity = dot product of normalized vectors
                emb1 = query_embedding / np.linalg.norm(query_embedding)
                emb2 = c["embedding"] / np.linalg.norm(c["embedding"])
                c["result"].similarity_score = max(0.0, float(np.dot(emb1, emb2)))

        reranked = self._apply_bm25_rerank(query_text, combined_candidates)

        # ── Step 4: MMR diversity selection ───────────────────────────────────
        return self._apply_mmr(reranked, top_k)

    # ─── Private: BM25 ───────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Custom BM25 tokenizer for extreme robustness:
        - Replaces punctuation with spaces
        - Merges adjacent letters and numbers (gox 110 -> gox110)
        - Generates 4-grams for long words to handle typos (transglutaminase -> rans, ansg...)
        """
        import re
        t = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = t.split()
        extended = list(tokens)
        
        # Merge adjacent letters and numbers for product codes
        for i in range(len(tokens) - 1):
            if (tokens[i].isalpha() and tokens[i+1].isdigit()) or \
               (tokens[i].isdigit() and tokens[i+1].isalpha()):
                extended.append(tokens[i] + tokens[i+1])
                
        # Generate 4-grams for long string typo-robustness
        for t_word in tokens:
            if len(t_word) > 4:
                for idx in range(len(t_word) - 3):
                    extended.append(t_word[idx:idx+4])
                    
        return extended

    def _build_bm25(self, ids: List[str], texts: List[str]) -> None:
        """Build BM25 index from scratch over the given corpus."""
        self._bm25_ids   = ids
        self._bm25_texts = texts
        self._bm25       = BM25Okapi([self._tokenize(text) for text in texts])
        print(f"[ChromaStore] BM25 index built over {len(texts)} fragments.")

    def _rebuild_bm25_from_store(self) -> None:
        """
        Reconstruct the in-memory BM25 index from persisted ChromaDB documents.
        Called on startup when an existing index is found on disk.
        Ensures keyword search works immediately without re-indexing.
        """
        results = self._collection.get(
            include = ["documents"],
            where   = {"source_document": {"$ne": METADATA_SENTINEL_ID}},
        )
        ids   = results["ids"]
        texts = results["documents"]

        if ids:
            self._build_bm25(ids, texts)
            print(f"[ChromaStore] BM25 index restored from {len(ids)} persisted fragments.")

    def _apply_bm25_rerank(
        self,
        query_text: str,
        candidates: List[dict],
    ) -> List[dict]:
        """
        Re-score candidates by combining semantic + BM25 scores.

        Formula: combined = α * semantic + (1-α) * bm25_norm
        where α = self._semantic_weight

        Semantic scores are already [0, 1] from ChromaDB.
        BM25 scores are normalized relative to the current candidate pool.
        """
        if self._bm25 is None or not candidates:
            return candidates

        query_tokens = self._tokenize(query_text)

        # Build a fast lookup: fragment_id → bm25 raw score
        all_bm25_scores = dict(zip(
            self._bm25_ids,
            self._bm25.get_scores(query_tokens),
        ))

        # Get raw BM25 scores for candidates
        candidate_bm25 = np.array([
            all_bm25_scores.get(c["result"].fragment.fragment_id, 0.0)
            for c in candidates
        ])

        # Normalize BM25 scores to [0, 1] relative to the pool
        bm25_max = candidate_bm25.max()
        bm25_norm = candidate_bm25 / bm25_max if bm25_max > 0 else candidate_bm25

        # Combine scores. Semantic scores are ALREADY [0, 1] from _map_to_candidates.
        for i, candidate in enumerate(candidates):
            semantic_score = candidate["result"].similarity_score
            hybrid_score = (
                self._semantic_weight       * float(semantic_score)
                + (1 - self._semantic_weight) * float(bm25_norm[i])
            )
            candidate["result"] = SearchResult(
                fragment         = candidate["result"].fragment,
                similarity_score = hybrid_score,
            )

        return sorted(candidates, key=lambda c: c["result"].similarity_score, reverse=True)

    # ─── Private: MMR ────────────────────────────────────────────────────────

    def _apply_mmr(
        self,
        candidates: List[dict],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance over the reranked candidate pool.

        Iteratively picks the candidate maximizing:
            MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Guarantees diversity: near-duplicate fragments are penalized.
        """
        if not candidates:
            return []

        # No min-max normalization of relevance scores — preserves absolute quality.
        # Reranked scores from BM25 are already in [0, 1].
        relevance_scores = np.array([c["result"].similarity_score for c in candidates])

        selected_indices:    List[int]       = []
        selected_embeddings: List[np.ndarray] = []

        for _ in range(min(top_k, len(candidates))):
            best_idx = None
            best_mmr = -np.inf

            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue

                relevance = float(relevance_scores[i])

                if selected_embeddings:
                    selected_matrix = np.stack(selected_embeddings)
                    redundancy = float(
                        np.max(selected_matrix @ candidate["embedding"])
                    )
                    redundancy = max(0.0, redundancy)
                else:
                    redundancy = 0.0

                mmr_score = (
                    self._mmr_lambda       * relevance
                    - (1 - self._mmr_lambda) * redundancy
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            selected_embeddings.append(candidates[best_idx]["embedding"])

        return [candidates[i]["result"] for i in selected_indices]

    # ─── Private: ChromaDB Helpers ────────────────────────────────────────────

    def _map_to_candidates(self, chroma_results: dict) -> List[dict]:
        """
        Map raw ChromaDB query output to candidate dicts.
        Each dict contains a SearchResult and the raw embedding for MMR.
        Sentinel documents are silently excluded.
        """
        candidates = []

        for fid, text, metadata, distance, embedding in zip(
            chroma_results["ids"][0],
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["distances"][0],
            chroma_results["embeddings"][0],
        ):
            if fid == METADATA_SENTINEL_ID:
                continue

            candidates.append({
                "result": SearchResult(
                    fragment=Fragment(
                        fragment_id     = fid,
                        source_document = metadata.get("source_document", "unknown"),
                        text            = text,
                        page_number     = metadata.get("page_number", 1),
                    ),
                    similarity_score = max(0.0, 1.0 - distance),
                ),
                "embedding": np.array(embedding, dtype=np.float32),
            })

        return candidates

    def _real_fragment_count(self) -> int:
        """
        Return fragment count excluding the metadata sentinel document.
        ChromaDB's .count() includes the sentinel, which skews display values.
        """
        total = self._collection.count()
        return max(0, total - 1) if self._sentinel_exists() else total

    def _sentinel_exists(self) -> bool:
        """Check whether the metadata sentinel document has been written."""
        try:
            result = self._collection.get(ids=[METADATA_SENTINEL_ID])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def _get_metadata_value(self, key: str) -> Optional[str]:
        """Retrieve a single value from the sentinel document's metadata."""
        try:
            result = self._collection.get(
                ids     = [METADATA_SENTINEL_ID],
                include = ["metadatas"],
            )
            if result["metadatas"]:
                return result["metadatas"][0].get(key)
        except Exception:
            pass
        return None