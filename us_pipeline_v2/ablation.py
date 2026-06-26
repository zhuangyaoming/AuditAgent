"""Ablation experiment presets for US Pipeline V2.

Four standard ablation variants for measuring component contributions:

  - NO_PRIOR:   Replace 15 prior subjects with all ~65 accounting categories.
  - NO_HYBRID:  Use dense retrieval only (disable BM25 + section extraction).
  - NO_MULTI_EXPERT: Single LLM call directly on retrieved text (no single/cross experts).
  - NO_CROSS_DOC: Disable cross-report trend analysis agent.

Plus a FULL baseline config with all components enabled.
"""

from config import PipelineConfig
from prior_subjects import get_us_prior_15, get_cn_prior_15, get_cn_us_prior, load_all_categories


def baseline() -> PipelineConfig:
    """Full pipeline with all components enabled (default config)."""
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_us_prior_15(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        single_expert_enabled=True,
        cross_expert_enabled=True,
        multi_expert_enabled=True,
    )


def no_prior() -> PipelineConfig:
    """Replace 15 prior subjects with all ~65 accounting categories.

    Tests whether the Bayesian-informed prior adds value over
    exhaustive category coverage. Uses smaller top-k to keep token
    budget manageable with 71 subjects.
    """
    return PipelineConfig(
        prior_enabled=False,
        prior_subjects=load_all_categories(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        bm25_top_k=2,
        dense_top_k=2,
        section_max_chars=1500,
        single_expert_enabled=True,
        cross_expert_enabled=True,
        multi_expert_enabled=True,
    )


def no_hybrid() -> PipelineConfig:
    """Dense retrieval only — disable BM25 and section extraction.

    Tests whether the multi-path hybrid retrieval (BM25 + dense + section)
    outperforms dense-only semantic search.
    """
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_us_prior_15(),
        bm25_enabled=False,
        dense_enabled=True,
        section_enabled=False,
        single_expert_enabled=True,
        cross_expert_enabled=True,
        multi_expert_enabled=True,
    )


def no_multi_expert() -> PipelineConfig:
    """Single LLM directly on retrieved text — no multi-expert agents.

    Tests whether the two-phase expert design (single-report + cross-report)
    outperforms a single-pass LLM analysis.
    """
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_us_prior_15(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        single_expert_enabled=False,
        cross_expert_enabled=False,
        multi_expert_enabled=False,
    )


def no_cross_doc() -> PipelineConfig:
    """Disable cross-report trend analysis agent.

    Tests whether cross-document temporal analysis adds signal
    beyond per-report analysis alone.
    """
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_us_prior_15(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        single_expert_enabled=True,
        cross_expert_enabled=False,
        multi_expert_enabled=True,
    )


def cn_to_us() -> PipelineConfig:
    """CN market 15 prior subjects → US EDGAR reports.

    Tests whether CN-market priors (Chinese accounting subjects translated to
    English keywords) have predictive power on US fraud cases.
    """
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_cn_prior_15(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        single_expert_enabled=True,
        cross_expert_enabled=True,
        multi_expert_enabled=True,
    )


def cn_us_to_us() -> PipelineConfig:
    """Combined CN+US prior subjects (~25) → US EDGAR reports.

    Tests whether adding CN-market priors expands coverage beyond US priors alone.
    Merged list deduplicates overlapping categories (e.g., Inventory, AR).
    """
    return PipelineConfig(
        prior_enabled=True,
        prior_subjects=get_cn_us_prior(),
        bm25_enabled=True,
        dense_enabled=True,
        section_enabled=True,
        single_expert_enabled=True,
        cross_expert_enabled=True,
        multi_expert_enabled=True,
    )


ABLATION_PRESETS: dict[str, callable] = {
    "baseline": baseline,
    "no_prior": no_prior,
    "no_hybrid": no_hybrid,
    "no_multi_expert": no_multi_expert,
    "no_cross_doc": no_cross_doc,
    "cn_to_us": cn_to_us,
    "cn_us_to_us": cn_us_to_us,
}
