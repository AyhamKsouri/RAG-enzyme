# src/infrastructure/document_loader.py

import re
import uuid
import csv
from typing import List, Optional
from pathlib import Path

from src.domain.models import Fragment


# Larger chunks → cross-column content stays together in one fragment
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100

# ── Boilerplate detector ──────────────────────────────────────────────────────
# Purposely conservative: only remove fragments that are PURELY boilerplate.
# Better to keep a borderline fragment than to destroy product content.
BOILERPLATE_SIGNATURE_GROUPS = [
    # Full food safety block
    ["total plate count", "salmonella", "coliforms at 30"],
    # Heavy metals only block
    ["cadmium", "mercury", "arsenic", "lead: <"],
    # GMO + allergen regulatory block
    ["gmo status", "1829/2003", "ionization status"],
]


def _is_boilerplate_fragment(text: str) -> bool:
    """
    Conservative boilerplate check — requires tight multi-term co-occurrence.
    Only pure regulatory blocks are removed; mixed content is always kept.
    """
    lowered = text.lower()
    return any(
        all(sig in lowered for sig in group)
        for group in BOILERPLATE_SIGNATURE_GROUPS
    )


class DocumentLoader:
    """
    Loads documents from a directory and splits them into overlapping
    fragments for semantic indexing.

    Design principles:
    - Large chunks (800 chars) keep multi-column PDF content coherent
    - Conservative boilerplate filter — never removes product content
    - No title-injection hack — real content must be preserved instead
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load_directory(self, directory_path: str) -> List[Fragment]:
        data_dir = Path(directory_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {directory_path}")

        all_fragments: List[Fragment] = []

        for file_path in sorted(data_dir.rglob("*")):
            fragments = self.load_file(file_path)
            if fragments:
                all_fragments.extend(fragments)
                print(f"[DocumentLoader] Loaded {len(fragments)} fragments from {file_path.name}")

        print(f"[DocumentLoader] Total fragments loaded: {len(all_fragments)}")
        return all_fragments

    def load_file(self, file_path: Path) -> List[Fragment]:
        """
        Load a single file (PDF, TXT, MD, CSV) and return its fragments.
        Returns an empty list if the file type is unsupported or extraction fails.
        """
        if file_path.suffix.lower() in {".txt", ".md"}:
            return self._load_text_file(file_path)
        elif file_path.suffix.lower() == ".pdf":
            return self._load_pdf_file(file_path)
        elif file_path.suffix.lower() == ".csv":
            return self._load_csv_file(file_path)
        return []

    # ─── Private: File Loaders ────────────────────────────────────────────────

    def _load_text_file(self, file_path: Path) -> List[Fragment]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self._process_text(text, source=file_path.name)

    def _load_pdf_file(self, file_path: Path) -> List[Fragment]:
        """
        Extract text from PDF while tracking page numbers.
        Returns fragments with correct page metadata.
        """
        pages_content = self._extract_pdf_pages(file_path)

        if not pages_content:
            return []

        all_fragments: List[Fragment] = []
        for page_num, text in pages_content:
            if not text or len(text.strip()) < 50:
                continue
                
            cleaned = self._clean_text(text)
            page_fragments = self._split_into_fragments(cleaned, source=file_path.name, page_number=page_num)
            all_fragments.extend(page_fragments)

        return all_fragments

    def _extract_pdf_pages(self, file_path: Path) -> List[tuple[int, str]]:
        """
        Extract text page-by-page from PDF.
        Returns a list of (page_number, text) tuples.
        """
        # Try pdfplumber first
        pages = self._extract_pages_pdfplumber(file_path)
        
        # Fallback to PyMuPDF
        if not pages:
            pages = self._extract_pages_pymupdf(file_path)
            
        return pages

    def _extract_pages_pdfplumber(self, file_path: Path) -> List[tuple[int, str]]:
        try:
            import pdfplumber
        except ImportError:
            return []
        try:
            pages = []
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        pages.append((i + 1, text))
            return pages
        except Exception as error:
            print(f"[DocumentLoader] pdfplumber error on {file_path.name}: {error}")
            return []

    def _extract_pages_pymupdf(self, file_path: Path) -> List[tuple[int, str]]:
        try:
            import fitz
        except ImportError:
            return []
        try:
            pages = []
            with fitz.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf):
                    text = page.get_text()
                    if text:
                        pages.append((i + 1, text))
            return pages
        except Exception as error:
            print(f"[DocumentLoader] PyMuPDF error on {file_path.name}: {error}")
            return []

    def _load_csv_file(self, file_path: Path) -> List[Fragment]:
        fragments = []
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                row_text = " | ".join(cell.strip() for cell in row if cell.strip())
                if len(row_text) > 20:
                    fragments.append(Fragment(
                        fragment_id=str(uuid.uuid4()),
                        source_document=file_path.name,
                        text=self._clean_text(row_text),
                        page_number=1, # CSVs don't have pages
                    ))
        return fragments

    # ─── Private: PDF Extractors ──────────────────────────────────────────────

    def _extract_pdf_pdfplumber(self, file_path: Path) -> tuple[str, Optional[str]]:
        """Deprecated: use _extract_pdf_pages instead."""
        try:
            import pdfplumber
        except ImportError:
            return "", "ImportError"
        try:
            parts = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        parts.append(text)
            return "\n".join(parts), None
        except Exception as error:
            print(f"[DocumentLoader] pdfplumber error on {file_path.name}: {error}")
            return "", "ExtractionError"

    def _extract_pdf_pymupdf(self, file_path: Path) -> tuple[str, Optional[str]]:
        """Deprecated: use _extract_pdf_pages instead."""
        try:
            import fitz
        except ImportError:
            return "", "ImportError"
        try:
            parts = []
            with fitz.open(str(file_path)) as pdf:
                for page in pdf:
                    parts.append(page.get_text())
            return "\n".join(parts), None
        except Exception as error:
            print(f"[DocumentLoader] PyMuPDF error on {file_path.name}: {error}")
            return "", "ExtractionError"

    # ─── Private: Text Processing ─────────────────────────────────────────────

    def _process_text(self, text: str, source: str, page_number: int = 1) -> List[Fragment]:
        """Clean → chunk."""
        cleaned = self._clean_text(text)
        fragments = self._split_into_fragments(cleaned, source, page_number)
        return fragments

    def _split_into_fragments(self, text: str, source: str, page_number: int = 1) -> List[Fragment]:
        fragments = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end].strip()

            if len(chunk) > 150:
                fragments.append(Fragment(
                    fragment_id=str(uuid.uuid4()),
                    source_document=source,
                    text=chunk,
                    page_number=page_number,
                ))

            start += self._chunk_size - self._chunk_overlap

        return fragments

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and strip repetitive company headers."""
        # Strip the massive repetitive company header found in the PDFs
        # We look for explicit website prefixes because of the email address fo@vtrbeyond.com
        text = re.sub(r"VTR&beyond.*?(?:www\.|gw\.)vtrbeyond\.com", "", text, flags=re.DOTALL)
        text = re.sub(r"Zhuhai.*?(?:www\.|gw\.)vtrbeyond\.com", "", text, flags=re.DOTALL)

        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[^\x20-\x7EÀ-ÿ\n]", " ", text)
        return text.strip()