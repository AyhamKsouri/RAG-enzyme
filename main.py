# main.py

import sys
from src.infrastructure.document_loader import DocumentLoader
from src.infrastructure.embedding_engine import SentenceTransformerEngine
from src.infrastructure.chroma_store import ChromaVectorStore
from src.infrastructure.file_hasher import compute_directory_hashes
from src.application.search_service import SemanticSearchService
from src.interface.cli import (
    display_welcome_banner,
    display_indexing_status,
    prompt_for_query,
    display_results,
    display_error,
    ask_continue,
)


DATA_DIRECTORY = "data"
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
TOP_K_RESULTS = 3


def main() -> None:
    display_welcome_banner()

    # ── 1. Initialize infrastructure ─────────────────────────────────────────
    embedding_engine = SentenceTransformerEngine()

    try:
        vector_store = ChromaVectorStore(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_model_name=embedding_engine.model_name,
        )
    except RuntimeError as error:
        display_error(str(error))
        sys.exit(1)

    search_service = SemanticSearchService(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        top_k=TOP_K_RESULTS,
    )

    # ── 2. Incremental indexing decision ─────────────────────────────────────
    if not vector_store.is_ready():
        # First run or model changed — full reindex
        _build_full_index(search_service, vector_store)

    else:
        # Store has data — check for new or modified files
        current_hashes = compute_directory_hashes(DATA_DIRECTORY)
        stored_hashes = vector_store.get_indexed_file_hashes()
        new_or_changed = {
            name: h
            for name, h in current_hashes.items()
            if stored_hashes.get(name) != h
        }

        if new_or_changed:
            print(
                f"[Main] Detected {len(new_or_changed)} new/modified file(s):\n"
                + "\n".join(f"  + {name}" for name in sorted(new_or_changed))
            )
            _build_incremental_index(
                search_service, vector_store, new_or_changed
            )
            # Save updated full hash map
            vector_store.save_index_metadata(current_hashes)

        else:
            print("[Main] Index is up to date — skipping document loading. ✓")
            search_service.mark_as_ready()

    display_indexing_status(vector_store._collection.count())

    # ── 3. Interactive search loop ────────────────────────────────────────────
    while True:
        query = prompt_for_query()
        try:
            results = search_service.search(query)
            display_results(query, results)
        except ValueError as error:
            display_error(str(error))

        if not ask_continue():
            break


def _build_full_index(
    service: SemanticSearchService,
    store: ChromaVectorStore,
) -> None:
    """Load all documents, encode, and persist from scratch."""
    print("[Main] No valid index found — performing full index build...")
    loader = DocumentLoader()

    try:
        fragments = loader.load_directory(DATA_DIRECTORY)
    except FileNotFoundError as error:
        display_error(str(error))
        sys.exit(1)

    if not fragments:
        display_error(f"No supported documents found in '{DATA_DIRECTORY}/'.")
        sys.exit(1)

    service.build_index(fragments)

    # Persist file hashes so next run can detect changes
    file_hashes = compute_directory_hashes(DATA_DIRECTORY)
    store.save_index_metadata(file_hashes)


def _build_incremental_index(
    service: SemanticSearchService,
    store: ChromaVectorStore,
    changed_files: dict[str, str],
) -> None:
    """Load and reindex only new or modified files."""
    from pathlib import Path
    loader = DocumentLoader()
    fragments = []

    for filename in sorted(changed_files.keys()):
        file_path = Path(DATA_DIRECTORY) / filename
        try:
            file_fragments = loader.load_file(file_path)
            if file_fragments:
                fragments.extend(file_fragments)
                print(f"[Main] Reindexed '{filename}': {len(file_fragments)} fragments")
        except Exception as error:
            print(f"[Main] ⚠ Failed to load '{filename}': {error}")

    if fragments:
        service.build_index(fragments)


if __name__ == "__main__":
    main()