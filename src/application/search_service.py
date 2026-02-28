# src/application/search_service.py 
 
from typing import List 
 
from src.domain.interfaces import EmbeddingPort, VectorStorePort 
from src.domain.models import Fragment, SearchResult 
 
 
class SemanticSearchService: 
     """ 
     Core use case: retrieve top-k relevant fragments for a natural language query. 
 
     Lifecycle: 
     - If store.is_ready() → skip build_index(), call search() directly 
     - If not ready       → call build_index(), then search() 
 
     This service never decides whether indexing is needed — 
     that decision belongs to main.py (composition root). 
     """ 
 
     def __init__( 
         self, 
         embedding_engine: EmbeddingPort, 
         vector_store: VectorStorePort, 
         top_k: int = 3, 
     ): 
         self._embedding_engine = embedding_engine 
         self._vector_store = vector_store 
         self._top_k = top_k 
         self._is_indexed = False 
 
     def build_index(self, fragments: List[Fragment]) -> None: 
         """Encode all fragments and persist to the vector store.""" 
         if not fragments: 
             raise ValueError("Fragment list is empty — nothing to index.") 
 
         print(f"[SearchService] Encoding {len(fragments)} fragments...") 
         embeddings = self._embedding_engine.encode([f.text for f in fragments]) 
 
         for fragment, embedding in zip(fragments, embeddings): 
             fragment.embedding = embedding 
 
         self._vector_store.index_fragments(fragments) 
         self._is_indexed = True 
         print("[SearchService] Index built successfully.") 
 
     def mark_as_ready(self) -> None: 
         """ 
         Signal that the store already has valid indexed data. 
         Called by main.py when store.is_ready() returns True, 
         allowing search() to proceed without build_index(). 
         """ 
         self._is_indexed = True 
 
     def search(self, query: str) -> List[SearchResult]: 
         if not self._is_indexed: 
             raise RuntimeError("Index not built. Call build_index() first.") 
 
         query = query.strip() 
         if not query: 
             raise ValueError("Query cannot be empty.") 
 
         query_embedding = self._embedding_engine.encode_single(query) 
 
         return self._vector_store.search( 
             query_text=query, 
             query_embedding=query_embedding, 
             top_k=self._top_k, 
         )
