"""Direct section extraction via regex pattern matching on financial statement text.

Reuses the regex-based TERM_MAPPING approach from us_adapter.py,
extracting the actual text surrounding keyword matches in item8.txt.
"""

import re
from pathlib import Path

from prior_subjects import build_search_patterns


def _extract_context(text: str, match_pos: int, window: int = 1500) -> str:
    """Extract symmetric context window around a regex match position."""
    start = max(0, match_pos - window)
    end = min(len(text), match_pos + window)
    # Try to break at paragraph boundaries
    chunk = text[start:end].strip()
    return chunk


def retrieve_sections(
    report_text: str,
    subjects: list[dict],
    max_chars: int = 3000,
    max_per_subject: int = 3,
) -> dict[str, list[dict]]:
    """Extract text sections around keyword matches for each prior subject.

    Args:
        report_text: Financial statement text (item8.txt or full_report.txt).
        subjects: List of {category, keywords} dicts.
        max_chars: Max characters per extracted context window.
        max_per_subject: Max number of context windows per subject.

    Returns:
        Dict mapping category → list of {text, score, index} results.
    """
    pattern = build_search_patterns(subjects)
    matches = list(pattern.finditer(report_text))
    if not matches:
        return {}

    # Build per-subject keyword → pattern mapping
    results: dict[str, list[dict]] = {}
    for subj in subjects:
        cat = subj["category"]
        cat_pattern = re.compile(
            "|".join(re.escape(kw) for kw in sorted(subj["keywords"], key=len, reverse=True)),
            re.IGNORECASE,
        )
        hits = []
        for m in matches:
            if cat_pattern.search(m.group()):
                ctx = _extract_context(report_text, m.start(), window=max_chars // 2)
                hits.append({
                    "text": ctx,
                    "score": 1.0,  # exact keyword match
                    "index": m.start(),
                })
            if len(hits) >= max_per_subject:
                break
        if hits:
            results[cat] = hits
    return results
