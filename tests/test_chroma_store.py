# tests/test_chroma_store.py

import numpy as np
import pytest
import chromadb

from src.domain.models import Fragment, SearchResult
from src.infrastructure.chroma_store import ChromaVectorStore, COLLECTION_NAME


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    """Fresh ChromaVectorStore backed by a temp directory for each test."""
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        embedding_model_name="test-model-v1",
    )


def _make_fragment(fid: str, text: str, embedding: np.ndarray) -> Fragment:
    f = Fragment(fragment_id=fid, source_document="test.pdf", text=text)
    f.embedding = embedding
    return f


def _unit_fragments(n: int) -> list:
    """Create n fragments with orthogonal unit vector embeddings."""
    embeddings = np.eye(n, dtype=np.float32)
    return [
        _make_fragment(f"f{i}", f"Fragment number {i}", embeddings[i])
        for i in range(n)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_is_ready_false_when_empty(store):
    assert store.is_ready() is False


def test_is_ready_true_after_indexing(store):
    fragments = _unit_fragments(3)
    store.index_fragments(fragments)
    assert store.is_ready() is True


def test_index_fragments_persists_correct_count(store):
    fragments = _unit_fragments(5)
    store.index_fragments(fragments)
    assert store._real_fragment_count() == 5


def test_search_returns_top_k_results(store):
    fragments = _unit_fragments(5)
    store.index_fragments(fragments)

    query_embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(
        query_text="test query",
        query_embedding=query_embedding,
        top_k=3,
    )

    assert len(results) == 3


def test_search_top_result_is_most_similar(store):
    fragments = _unit_fragments(4)
    store.index_fragments(fragments)

    # Query most similar to f0
    query_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(
        query_text="test",
        query_embedding=query_embedding,
        top_k=3,
    )

    assert results[0].fragment.fragment_id == "f0"
    assert results[0].similarity_score > results[1].similarity_score


def test_search_results_are_valid_domain_objects(store):
    fragments = _unit_fragments(3)
    store.index_fragments(fragments)

    query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search("query", query_embedding, top_k=2)

    for result in results:
        assert result.fragment.text is not None
        assert result.fragment.source_document == "test.pdf"
        assert 0.0 <= result.similarity_score <= 1.0


def test_upsert_is_idempotent(store):
    """Calling index_fragments twice should not duplicate entries."""
    fragments = _unit_fragments(4)
    store.index_fragments(fragments)
    store.index_fragments(fragments)  # Second call — should upsert, not duplicate
    assert store._real_fragment_count() == 4


def test_model_fingerprint_mismatch_triggers_reindex(tmp_path):
    """Changing the model name should invalidate the existing index."""
    persist_dir = str(tmp_path / "chroma_fingerprint_test")

    # First run with model v1
    store_v1 = ChromaVectorStore(
        persist_directory=persist_dir,
        embedding_model_name="model-v1",
    )
    store_v1.index_fragments(_unit_fragments(3))
    assert store_v1.is_ready() is True

    # Second run with model v2 — should detect mismatch
    store_v2 = ChromaVectorStore(
        persist_directory=persist_dir,
        embedding_model_name="model-v2",
    )
    assert store_v2.is_ready() is False


def test_empty_fragment_list_raises(store):
    with pytest.raises(ValueError, match="empty"):
        store.index_fragments([])


def test_fragment_missing_embedding_raises(store):
    fragment = Fragment(
        fragment_id="f_no_emb",
        source_document="doc.pdf",
        text="some text",
    )
    # embedding is None — not set
    with pytest.raises(ValueError, match="missing embeddings"):
        store.index_fragments([fragment])

def test_save_and_retrieve_file_hashes(store):
    """File hashes should persist and be retrievable across store instances."""
    fragments = _unit_fragments(3)
    store.index_fragments(fragments)

    hashes = {"doc1.pdf": "abc123", "doc2.pdf": "def456"}
    store.save_index_metadata(hashes)

    retrieved = store.get_indexed_file_hashes()
    assert retrieved == hashes


def test_get_indexed_file_hashes_empty_when_no_metadata(store):
    """Should return empty dict when no metadata has been saved."""
    result = store.get_indexed_file_hashes()
    assert result == {}


def test_init_raises_clean_error_on_bad_path():
    """Should raise RuntimeError with helpful message on bad persist path."""
    # Use a path that can't be a valid directory (file used as directory)
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False) as f:
        bad_path = f.name
    try:
        with pytest.raises(RuntimeError, match="Failed to initialize ChromaDB"):
            ChromaVectorStore(
                persist_directory=bad_path,
                embedding_model_name="test-model",
            )
    finally:
        os.unlink(bad_path)


def test_mmr_returns_diverse_results(store):
    """ChromaStore MMR should not return near-duplicate fragments."""
    # f0 and f1 are near-identical (both close to query)
    # f2 is moderately relevant but diverse
    f0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f1 = np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)
    f1 /= np.linalg.norm(f1)
    f2 = np.array([0.5, 0.866, 0.0, 0.0], dtype=np.float32)

    fragments = [
        _make_fragment("f0", "fragment zero", f0),
        _make_fragment("f1", "fragment one near duplicate", f1),
        _make_fragment("f2", "fragment two diverse", f2),
    ]
    fragments[0].source_document = "doc_a.pdf"
    fragments[1].source_document = "doc_a.pdf"
    fragments[2].source_document = "doc_b.pdf"

    store.index_fragments(fragments)

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search("test query", query, top_k=2)

    assert results[0].fragment.fragment_id == "f0"
    sources = {r.fragment.source_document for r in results}
    assert len(sources) == 2, "MMR should return results from different sources"



def test_bm25_rebuilt_on_new_store_instance(tmp_path):
    """
    BM25 index should be rebuilt from disk on a fresh store instance.
    Keyword search must work on the second run without re-indexing.
    """
    persist_dir = str(tmp_path / "chroma_bm25_rebuild")

    # First instance — index fragments
    store_a = ChromaVectorStore(
        persist_directory    = persist_dir,
        embedding_model_name = "test-model-v1",
    )
    fragments = _unit_fragments(4)
    store_a.index_fragments(fragments)
    store_a.save_index_metadata({"doc.pdf": "hash123"})

    # Second instance — simulate fresh process startup
    store_b = ChromaVectorStore(
        persist_directory    = persist_dir,
        embedding_model_name = "test-model-v1",
    )

    # BM25 must be available — _bm25 should not be None
    assert store_b._bm25 is not None, "BM25 index should be rebuilt from persisted data"


def test_hybrid_search_boosts_keyword_match(tmp_path):
    """
    BM25 should boost fragments containing exact query keywords
    above semantically similar but keyword-missing fragments.
    """
    persist_dir = str(tmp_path / "chroma_hybrid")
    store = ChromaVectorStore(
        persist_directory    = persist_dir,
        embedding_model_name = "test-model-v1",
        semantic_weight      = 0.5,  # Equal weight to make BM25 effect visible
    )

    # All fragments have identical embeddings — only BM25 can differentiate
    shared_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    exact_match = _make_fragment("f_exact", "GOX 110 dosage is 5-40 ppm", shared_embedding)
    no_match    = _make_fragment("f_none",  "enzyme preparation bakery product", shared_embedding)
    partial     = _make_fragment("f_part",  "GOX enzyme bakery application", shared_embedding)

    exact_match.source_document = "gox.pdf"
    no_match.source_document    = "other.pdf"
    partial.source_document     = "partial.pdf"

    store.index_fragments([exact_match, no_match, partial])

    query_embedding = shared_embedding.copy()
    results = store.search("GOX 110 dosage", query_embedding, top_k=3)

    # Fragment with exact keyword match must rank highest
    assert results[0].fragment.fragment_id == "f_exact", (
        "BM25 should rank exact keyword match first when semantic scores are equal"
    )


def test_real_fragment_count_excludes_sentinel(store):
    """_real_fragment_count() should not count the metadata sentinel."""
    fragments = _unit_fragments(5)
    store.index_fragments(fragments)
    store.save_index_metadata({"file.pdf": "hash"})

    # ChromaDB total = 6 (5 fragments + 1 sentinel)
    # _real_fragment_count() must return 5
    assert store._real_fragment_count() == 5


def test_search_pipeline_order_chromadb_bm25_mmr(store):
    """
    Full pipeline integration: ChromaDB → BM25 → MMR.
    Final results must be valid SearchResult objects with diverse sources.
    """
    embeddings = np.eye(6, dtype=np.float32)
    fragments  = [
        _make_fragment(f"f{i}", f"lipase enzyme fragment {i}", embeddings[i])
        for i in range(6)
    ]
    for i, f in enumerate(fragments):
        f.source_document = f"doc_{i % 3}.pdf"  # 3 sources, 2 fragments each

    store.index_fragments(fragments)

    query_embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search("lipase enzyme", query_embedding, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0.0 <= r.similarity_score <= 1.0 for r in results)    