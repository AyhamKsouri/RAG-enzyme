from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import shutil
import os
from pathlib import Path

from src.infrastructure.document_loader import DocumentLoader
from src.infrastructure.embedding_engine import SentenceTransformerEngine
from src.infrastructure.chroma_store import ChromaVectorStore
from src.infrastructure.file_hasher import compute_directory_hashes
from src.application.search_service import SemanticSearchService

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIRECTORY = "data"
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
DEFAULT_TOP_K = 3

def get_directory_size_mb(directory: str) -> float:
    """Calculate total size of a directory in Megabytes."""
    total_size = 0
    path = Path(directory)
    if not path.exists():
        return 0.0
    for f in path.rglob('*'):
        if f.is_file():
            total_size += f.stat().st_size
    return round(total_size / (1024 * 1024), 2)

# ── API Models ───────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K

class FragmentSchema(BaseModel):
    fragment_id: str
    source_document: str
    text: str
    page_number: int

class SearchResponse(BaseModel):
    query: str
    results: List[dict] # Simplified for direct JSON response

# ── App Initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Enzyme API",
    description="Semantic + Keyword search over technical documentation.",
    version="1.0.0"
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize infrastructure (global scope for singleton behavior)
embedding_engine = SentenceTransformerEngine()
vector_store = ChromaVectorStore(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_model_name=embedding_engine.model_name,
)
search_service = SemanticSearchService(
    embedding_engine=embedding_engine,
    vector_store=vector_store,
    top_k=DEFAULT_TOP_K,
)

# Auto-configure service state
if vector_store.is_ready():
    print("[API] Persistent index detected. Service is READY.")
    search_service.mark_as_ready()
else:
    print("[API] WARNING: Vector store is not ready. Please run main.py to index documents first.")

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "message": "Technical Document RAG API is running.",
        "status": "ready" if vector_store.is_ready() else "indexing_required",
        "fragments_indexed": vector_store._real_fragment_count()
    }

@app.get("/status")
def get_status():
    """Returns the current readiness of the vector store and indexing statistics."""
    return {
        "is_ready": vector_store.is_ready(),
        "fragments_indexed": vector_store._real_fragment_count(),
        "embedding_model": embedding_engine.model_name,
        "storage_used_mb": get_directory_size_mb(DATA_DIRECTORY)
    }

@app.get("/documents")
def get_documents():
    """Returns the list of indexed documents and their fragment counts."""
    try:
        stats = vector_store.get_document_stats()
        return {"documents": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def trigger_reindex():
    """Manually trigger an incremental re-index of the data directory."""
    try:
        current_hashes = compute_directory_hashes(DATA_DIRECTORY)
        stored_hashes = vector_store.get_indexed_file_hashes()
        
        # Find new or changed files
        new_or_changed = {
            name: h
            for name, h in current_hashes.items()
            if stored_hashes.get(name) != h
        }
        
        # Also find deleted files (in stored but not in current)
        deleted_files = [
            name for name in stored_hashes.keys()
            if name not in current_hashes
        ]
        
        # Handle deletions
        for filename in deleted_files:
            vector_store.delete_document(filename)
            print(f"[API] Auto-removed deleted file from index: {filename}")

        if new_or_changed:
            _build_incremental_index(search_service, vector_store, new_or_changed)
            vector_store.save_index_metadata(current_hashes)
            
        return {
            "message": "Re-indexing complete.",
            "processed": list(new_or_changed.keys()),
            "deleted": deleted_files,
            "is_ready": vector_store.is_ready()
        }
    except Exception as e:
        print(f"[API] Re-indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a document from both the filesystem and the vector index."""
    file_path = Path(DATA_DIRECTORY) / filename
    
    # 1. Delete from vector store (even if file is already gone)
    try:
        vector_store.delete_document(filename)
    except Exception as e:
        print(f"[API] Warning deleting {filename} from vector store: {e}")

    # 2. Delete physical file
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"[API] Deleted file: {file_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
    
    return {"message": f"Successfully deleted document '{filename}'"}

@app.get("/documents/{filename}/file")
def get_document_file(filename: str):
    """Serve a PDF/TXT/MD file from the data directory."""
    file_path = Path(DATA_DIRECTORY) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    media_type = "application/pdf"
    if file_path.suffix.lower() == ".txt":
        media_type = "text/plain"
    elif file_path.suffix.lower() == ".md":
        media_type = "text/markdown"
    elif file_path.suffix.lower() == ".csv":
        media_type = "text/csv"
        
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload PDF/TXT/MD files and trigger incremental indexing."""
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = Path(DATA_DIRECTORY) / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
    
    # Trigger incremental re-index
    try:
        current_hashes = compute_directory_hashes(DATA_DIRECTORY)
        stored_hashes = vector_store.get_indexed_file_hashes()
        
        new_or_changed = {
            name: h
            for name, h in current_hashes.items()
            if stored_hashes.get(name) != h
        }
        
        if new_or_changed:
            _build_incremental_index(search_service, vector_store, new_or_changed)
            vector_store.save_index_metadata(current_hashes)
            
        return {
            "message": f"Successfully uploaded {len(saved_files)} files.",
            "files": saved_files,
            "indexed_new": list(new_or_changed.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during indexing: {str(e)}")

def _build_incremental_index(
    service: SemanticSearchService,
    store: ChromaVectorStore,
    changed_files: dict[str, str],
) -> None:
    """Load and reindex only new or modified files."""
    loader = DocumentLoader()
    fragments = []

    for filename in sorted(changed_files.keys()):
        file_path = Path(DATA_DIRECTORY) / filename
        try:
            file_fragments = loader.load_file(file_path)
            if file_fragments:
                fragments.extend(file_fragments)
                print(f"[API] Reindexed '{filename}': {len(file_fragments)} fragments")
        except Exception as error:
            print(f"[API] ⚠ Failed to load '{filename}': {error}")

    if fragments:
        service.build_index(fragments)

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not vector_store.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="Search service is not ready. Documents must be indexed via main.py first."
        )
    
    try:
        # Override top_k if provided in request
        search_service._top_k = request.top_k
        
        results = search_service.search(request.query)
        
        # Map domain models to JSON-serializable dicts
        serializable_results = [
            {
                "fragment": {
                    "id": r.fragment.fragment_id,
                    "source": r.fragment.source_document,
                    "text": r.fragment.text,
                    "page_number": r.fragment.page_number
                },
                "score": round(float(r.similarity_score), 4)
            }
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=serializable_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
