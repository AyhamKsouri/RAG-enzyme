# src/domain/interfaces.py 
 
from abc import ABC, abstractmethod 
from typing import List 
import numpy as np 
 
from .models import Fragment, SearchResult 
 
 
class EmbeddingPort(ABC): 
    """ 
    Port for any embedding engine. 
    Intentionally minimal — no infrastructure concerns like model naming. 
    """ 
 
    @abstractmethod 
    def encode(self, texts: List[str]) -> np.ndarray: ... 
 
    @abstractmethod 
    def encode_single(self, text: str) -> np.ndarray: ... 
 
 
class VectorStorePort(ABC): 
 
    @abstractmethod 
    def index_fragments(self, fragments: List[Fragment]) -> None: ... 
 
    @abstractmethod 
    def search( 
        self, 
        query_text: str, 
        query_embedding: np.ndarray, 
        top_k: int, 
    ) -> List[SearchResult]: ... 
 
    @abstractmethod 
    def is_ready(self) -> bool: ... 
 
    @abstractmethod 
    def get_indexed_file_hashes(self) -> dict[str, str]: 
        """ 
        Return a mapping of { filename: hash } for all indexed source files. 
        Used by main.py to detect new or modified documents since last run. 
        """ 
        ... 

    @abstractmethod
    def get_document_stats(self) -> List[dict]:
        """
        Return a list of indexed documents with their fragment counts.
        """
        ...
