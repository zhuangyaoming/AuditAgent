"""BM25 sparse retrieval — keyword-based search over EDGAR report text."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple English tokenizer: lowercase + split on non-alpha."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _split_paragraphs(text: str, min_chars: int = 100) -> list[str]:
    """Split text into paragraphs (double-newline or sections)."""
    raw = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw if len(p.strip()) >= min_chars]
    if not paragraphs:
        paragraphs = [text]
    return paragraphs


class BM25Retriever:
    """BM25 sparse retriever for a single report text."""

    def __init__(self, report_text: str):
        self.paragraphs = _split_paragraphs(report_text)
        self.tokenized = [_tokenize(p) for p in self.paragraphs]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for paragraphs matching the query string.

        Returns list of {text, score, index}.
        """
        if not self.bm25:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = np.array(self.bm25.get_scores(tokenized_query))
        if scores.max() == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.paragraphs[idx],
                    "score": float(scores[idx]),
                    "index": int(idx),
                })
        return results


def retrieve_bm25(report_text: str, subjects: list[dict], top_k: int = 5) -> dict[str, list[dict]]:
    """Run BM25 retrieval for all prior subjects against one report.

    Args:
        report_text: Full text of the report (full_report.txt).
        subjects: List of {category, keywords} dicts.
        top_k: Number of top paragraphs to return per subject.

    Returns:
        Dict mapping category → list of {text, score, index} results.
    """
    retriever = BM25Retriever(report_text)
    results = {}
    for subj in subjects:
        query = " ".join(subj["keywords"])
        hits = retriever.search(query, top_k=top_k)
        if hits:
            results[subj["category"]] = hits
    return results
