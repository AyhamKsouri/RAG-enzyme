# src/infrastructure/embedding_engine.py 
# model_name stays as a concrete property — NOT from the abstract port 
 
import numpy as np 
from typing import List 
from sentence_transformers import SentenceTransformer 
 
from src.domain.interfaces import EmbeddingPort 
 
 
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2" 
 
 
class SentenceTransformerEngine(EmbeddingPort): 
 
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME): 
        print(f"[EmbeddingEngine] Loading model: {model_name} ...") 
        self._model_name = model_name 
        self._model = SentenceTransformer(model_name) 
        print(f"[EmbeddingEngine] Model ready.") 
 
    @property 
    def model_name(self) -> str: 
        """Concrete property — used by main.py for fingerprinting. Not part of port.""" 
        return self._model_name 
 
    def encode(self, texts: List[str]) -> np.ndarray: 
        return self._model.encode( 
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=True, 
            batch_size=32, 
            normalize_embeddings=True, 
        ) 
 
    def encode_single(self, text: str) -> np.ndarray: 
        return self._model.encode( 
            text, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
        )
