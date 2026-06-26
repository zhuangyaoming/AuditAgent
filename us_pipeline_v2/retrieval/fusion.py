"""Reciprocal Rank Fusion (RRF) for merging multi-path retrieval results.

score(d) = Σ_{path} 1 / (k + rank_i(d))

Where k=60 (standard RRF constant), rank_i(d) is the rank of document d
in retrieval path i (1-indexed).
"""

from collections import OrderedDict

# Fallback import for when retrieval modules are not available
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def _text_hash(text: str, length: int = 80) -> str:
    """Simple dedup hash from the start of the text."""
    return text[:length].strip().lower()


def rrf_fuse(
    path_results: list[dict[str, list[dict]]],
    k: int = 60,
) -> list[dict]:
    """Fuse results from multiple retrieval paths using Reciprocal Rank Fusion.

    Args:
        path_results: List of results dicts, each from one retrieval path.
                      Each dict is {category: [{text, score, index}, ...]}.
        k: RRF constant (default 60).

    Returns:
        Merged and ranked list of {text, score, sources} dicts,
        sorted by RRF score descending.
    """
    # Aggregate RRF scores per unique passage
    passage_scores: dict[str, float] = {}
    passage_sources: dict[str, list[str]] = {}
    passage_text: dict[str, str] = {}

    for path_idx, path_result in enumerate(path_results):
        for category, hits in path_result.items():
            for rank, hit in enumerate(hits, start=1):
                text = hit["text"].strip()
                h = _text_hash(text)
                rrf_contrib = 1.0 / (k + rank)
                passage_scores[h] = passage_scores.get(h, 0.0) + rrf_contrib
                if h not in passage_sources:
                    passage_sources[h] = []
                passage_sources[h].append(category)
                passage_text[h] = text

    # Sort by RRF score descending
    sorted_hashes = sorted(passage_scores.keys(), key=lambda h: passage_scores[h], reverse=True)

    fused = []
    for h in sorted_hashes:
        fused.append({
            "text": passage_text[h],
            "score": round(passage_scores[h], 6),
            "sources": list(OrderedDict.fromkeys(passage_sources[h])),
        })
    return fused


def fuse_and_concat(
    path_results: list[dict[str, list[dict]]],
    k: int = 60,
    max_tokens: int = 30000,
) -> str:
    """Fuse retrieval results and concatenate into a single token-truncated string.

    Args:
        path_results: Results from multiple retrieval paths.
        k: RRF constant.
        max_tokens: Maximum tokens in the output string.

    Returns:
        Concatenated text from fused retrieval results.
    """
    fused = rrf_fuse(path_results, k=k)
    if not fused:
        return ""

    # Build sections grouped by source category
    lines = []
    seen_texts = set()
    for item in fused:
        h = _text_hash(item["text"])
        if h in seen_texts:
            continue
        seen_texts.add(h)
        sources_str = ", ".join(item["sources"])
        lines.append(f"[{sources_str}] (RRF={item['score']:.4f})\n{item['text']}\n")

    result = "\n".join(lines)

    # Truncate by tokens if needed
    if _ENCODING and max_tokens > 0:
        tokens = _ENCODING.encode(result)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            result = _ENCODING.decode(tokens)

    return result
