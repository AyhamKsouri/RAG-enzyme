# tests/test_vector_store.py

import numpy as np
import pytest
from src.domain.models import Fragment
from src.infrastructure.vector_store import InMemoryVectorStore


def _make_fragment(fragment_id: str, text: str, embedding: np.ndarray) -> Fragment:
    f = Fragment(fragment_id=fragment_id, source_document="test.txt", text=text)
    f.embedding = embedding
    return f


def test_search_returns_top_k_results():
    store = InMemoryVectorStore()
    # Normalized embeddings (unit vectors)
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0],
    ], dtype=np.float32)

    fragments = [
        _make_fragment(f"f{i}", f"Fragment {i}", embeddings[i])
        for i in range(4)
    ]
    store.index_fragments(fragments)

    query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_text="Fragment 0", query_embedding=query_emb, top_k=3)

    assert len(results) == 3
    assert results[0].fragment.fragment_id == "f0"  # Most similar
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)


def test_search_sorted_descending():
    store = InMemoryVectorStore()
    embeddings = np.eye(3, dtype=np.float32)
    fragments = [_make_fragment(f"f{i}", f"text {i}", embeddings[i]) for i in range(3)]
    store.index_fragments(fragments)

    query_emb = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    results = store.search(query_text="query", query_embedding=query_emb, top_k=3)

    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_raises():
    store = InMemoryVectorStore()
    with pytest.raises(RuntimeError):
        store.search(query_text="query", query_embedding=np.array([1.0, 0.0]), top_k=3)

def test_mmr_returns_diverse_sources():
    """MMR should pick a diverse second result over a near-duplicate."""
    store = InMemoryVectorStore(use_mmr=True, mmr_lambda=0.6)

    # All embeddings are pre-normalized unit vectors
    # f0: perfectly matches query [1, 0, 0, 0]
    # f1: near-duplicate of f0 (very similar to already-selected)
    # f2: moderately relevant AND diverse from f0 → MMR should prefer this
    # f3: irrelevant
    f0_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f1_emb = np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)
    f1_emb /= np.linalg.norm(f1_emb)  # normalize
    f2_emb = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    f2_emb /= np.linalg.norm(f2_emb)  # normalize — relevance=0.6, diverse from f0
    f3_emb = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal = irrelevant

    fragments = [
        _make_fragment("f0", "text 0", f0_emb),
        _make_fragment("f1", "text 1", f1_emb),
        _make_fragment("f2", "text 2", f2_emb),
        _make_fragment("f3", "text 3", f3_emb),
    ]
    fragments[0].source_document = "doc_a"
    fragments[1].source_document = "doc_a"  # near-duplicate of f0
    fragments[2].source_document = "doc_b"  # diverse + moderately relevant
    fragments[3].source_document = "doc_c"  # irrelevant

    store.index_fragments(fragments)

    query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_text="query", query_embedding=query_emb, top_k=2)

    sources = [r.fragment.source_document for r in results]
    assert results[0].fragment.fragment_id == "f0", "Most relevant should still be first"
    assert len(set(sources)) == 2, "MMR should pick diverse doc_b over near-duplicate doc_a"