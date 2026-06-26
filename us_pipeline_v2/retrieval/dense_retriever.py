"""Dense retrieval using sentence-transformers for semantic passage search."""

import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Lightweight model — downloaded once and cached
MODEL_NAME = "all-MiniLM-L6-v2"

_encoder: Optional[SentenceTransformer] = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def _split_passages(text: str, min_chars: int = 100, max_chars: int = 2000) -> list[str]:
    """Split text into overlapping passages suitable for dense encoding."""
    paragraphs = re.split(r"\n\s*\n", text)
    passages = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < min_chars:
            continue
        if len(para) <= max_chars:
            passages.append(para)
        else:
            # Split long paragraphs into overlapping chunks
            words = para.split()
            chunk_size = 200
            overlap = 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk) >= min_chars:
                    passages.append(chunk)
    if not passages:
        passages = [text[:max_chars]]
    return passages


class DenseRetriever:
    """Dense (semantic) retriever for a single report text."""

    def __init__(self, report_text: str):
        self.passages = _split_passages(report_text)
        self.encoder = _get_encoder()
        self._embeddings: Optional[np.ndarray] = None

    def _encode(self) -> np.ndarray:
        if self._embeddings is None and self.passages:
            self._embeddings = self.encoder.encode(
                self.passages,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return self._embeddings if self._embeddings is not None else np.array([])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for passages semantically similar to the query.

        Returns list of {text, score, index}.
        """
        embeddings = self._encode()
        if len(embeddings) == 0:
            return []
        query_embedding = self.encoder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False,
        )
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # minimum similarity threshold
                results.append({
                    "text": self.passages[idx],
                    "score": float(similarities[idx]),
                    "index": int(idx),
                })
        return results


def retrieve_dense(report_text: str, subjects: list[dict], top_k: int = 5) -> dict[str, list[dict]]:
    """Run dense retrieval for all prior subjects against one report.

    Args:
        report_text: Full text of the report (full_report.txt).
        subjects: List of {category, keywords} dicts.
        top_k: Number of top passages to return per subject.

    Returns:
        Dict mapping category → list of {text, score, index} results.
    """
    retriever = DenseRetriever(report_text)
    results = {}
    for subj in subjects:
        # Use category name + first 3 keywords as query
        query = f"{subj['category']}: " + ", ".join(subj["keywords"][:3])
        hits = retriever.search(query, top_k=top_k)
        if hits:
            results[subj["category"]] = hits
    return results
