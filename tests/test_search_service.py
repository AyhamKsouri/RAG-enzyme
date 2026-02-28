# tests/test_search_service.py

import numpy as np
import pytest
from unittest.mock import MagicMock
from src.domain.models import Fragment, SearchResult
from src.application.search_service import SemanticSearchService


def _make_mock_engine(embedding_value: np.ndarray):
    engine = MagicMock()
    engine.encode.return_value = np.stack([embedding_value] * 5)
    engine.encode_single.return_value = embedding_value
    return engine


def _make_mock_store(results):
    store = MagicMock()
    store.search.return_value = results
    return store


def test_search_requires_index():
    service = SemanticSearchService(MagicMock(), MagicMock())
    with pytest.raises(RuntimeError, match="Index not built"):
        service.search("test query")


def test_search_raises_on_empty_query():
    engine = _make_mock_engine(np.array([0.1, 0.2, 0.3]))
    store = _make_mock_store([])

    service = SemanticSearchService(engine, store)
    fragments = [Fragment(f"f{i}", "doc.txt", f"text {i}") for i in range(3)]
    service.build_index(fragments)

    with pytest.raises(ValueError, match="empty"):
        service.search("   ")


def test_build_index_assigns_embeddings():
    dummy_embedding = np.array([0.1, 0.2, 0.3])
    engine = _make_mock_engine(dummy_embedding)
    store = MagicMock()

    service = SemanticSearchService(engine, store)
    fragments = [Fragment(f"f{i}", "doc.txt", f"text {i}") for i in range(3)]
    service.build_index(fragments)

    assert all(f.embedding is not None for f in fragments)
    store.index_fragments.assert_called_once()