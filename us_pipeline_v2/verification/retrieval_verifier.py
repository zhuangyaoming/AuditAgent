"""Retrieval quality verification — keyword hit rates, empty retrieval warnings.

Runs after Phase 1 retrieval to ensure search quality is traceable.
Results are written as extra columns in Excel output.
"""

import logging
from collections import Counter

from config import PipelineConfig
from prior_subjects import US_PRIOR_15


def verify_retrieval(
    report_data: list[dict],
    config: PipelineConfig,
) -> dict:
    """Verify retrieval quality across all reports.

    Checks:
      1. Per-subject keyword hit rate in retrieved passages.
      2. Empty retrieval warnings (no passages found for a subject).
      3. Path coverage stats (how many paths contributed).

    Args:
        report_data: List of {report_name, report_path, retrieved_text, path_results}.
        config: PipelineConfig with prior_subjects.

    Returns:
        Dict with verification stats (also logged).
    """
    subjects = config.prior_subjects
    all_keywords: dict[str, list[str]] = {}
    for s in subjects:
        all_keywords[s["category"]] = s["keywords"]

    total_passages = 0
    total_chars = 0
    path_contributions: Counter = Counter()
    subject_hit_count: Counter = Counter()
    empty_subjects: list[str] = []
    per_report_stats: list[dict] = []

    for rd in report_data:
        report_name = rd["report_name"]
        fused_text = rd.get("retrieved_text", "")
        n_chars = len(fused_text)
        total_chars += n_chars

        # Count passages (separated by double-newline blocks)
        passages = [p for p in fused_text.split("\n\n") if p.strip()]
        n_passages = len(passages)
        total_passages += n_passages

        # Count path contributions
        for path_res in rd.get("path_results", []):
            for cat, hits in path_res.items():
                path_contributions[cat] += len(hits)

        # Per-subject keyword hit check
        fused_lower = fused_text.lower()
        hits_per_subject = {}
        for cat, keywords in all_keywords.items():
            hits = 0
            for kw in keywords:
                if kw.lower() in fused_lower:
                    hits += 1
            hits_per_subject[cat] = hits
            if hits > 0:
                subject_hit_count[cat] += 1

        # Check for empty subjects
        for cat, hits in hits_per_subject.items():
            if hits == 0:
                empty_subjects.append(f"{report_name}/{cat}")

        per_report_stats.append({
            "report": report_name,
            "chars": n_chars,
            "passages": n_passages,
            "hit_subjects": sum(1 for h in hits_per_subject.values() if h > 0),
            "total_subjects": len(subjects),
        })

    # Log summary
    n_reports = len(report_data)
    n_subjects = len(subjects)
    subjects_with_hits = len(subject_hit_count)

    logging.info(
        f"Retrieval Verification: {n_reports} reports, {n_subjects} subjects, "
        f"{total_passages} passages, {total_chars} chars"
    )
    logging.info(f"  Subjects with >=1 keyword hit: {subjects_with_hits}/{n_subjects}")

    if empty_subjects:
        logging.warning(f"  Empty subjects: {len(empty_subjects)} — first 10: {empty_subjects[:10]}")

    hit_rates = {
        cat: round(subject_hit_count.get(cat, 0) / n_reports, 3) if n_reports > 0 else 0.0
        for cat in all_keywords
    }

    zero_hit = [cat for cat, rate in hit_rates.items() if rate == 0.0]
    if zero_hit:
        logging.warning(f"  Zero-hit subjects (no reports matched): {zero_hit}")

    stats = {
        "n_reports": n_reports,
        "n_subjects": n_subjects,
        "total_passages": total_passages,
        "total_chars": total_chars,
        "subjects_with_hits": subjects_with_hits,
        "hit_rates": hit_rates,
        "empty_subjects": empty_subjects,
        "zero_hit_subjects": zero_hit,
        "path_contributions": dict(path_contributions),
        "per_report": per_report_stats,
    }
    return stats
