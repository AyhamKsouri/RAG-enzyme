# src/domain/models.py

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Fragment:
    """
    Represents a single searchable unit of text extracted from a document.
    """
    fragment_id: str
    source_document: str
    text: str
    page_number: int = 1
    embedding: np.ndarray = field(default=None, repr=False)


@dataclass
class SearchResult:
    """
    Represents a ranked search result returned to the user.
    """
    fragment: Fragment
    similarity_score: float

    def __repr__(self) -> str:
        preview = self.fragment.text[:80].replace("\n", " ")
        return (
            f"SearchResult(score={self.similarity_score:.4f}, "
            f"source='{self.fragment.source_document}', "
            f"preview='{preview}...')"
        )