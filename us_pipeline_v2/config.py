"""Global configuration for US Pipeline V2."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prior_subjects import get_us_prior_15


@dataclass
class PipelineConfig:
    # ---- Paths ----
    edgar_text_base: str = "d:/mainfiles/AuditAgent/finfraud_processing/data/edgar_reports_text"
    cases_base: str = "d:/mainfiles/AuditAgent/finfraud_processing/data/processed/cases"
    classification_file: str = "d:/mainfiles/AuditAgent/finfraud_processing/data/processed/cases/account_classification.json"
    output_dir: str = "d:/mainfiles/AuditAgent/us_pipeline_v2/results"
    report_dir: str = "d:/mainfiles/AuditAgent/us_pipeline_v2/reports"

    # ---- Prior ----
    prior_enabled: bool = True
    prior_subjects: list = field(default_factory=get_us_prior_15)

    # ---- Retrieval switches ----
    bm25_enabled: bool = True
    dense_enabled: bool = True
    section_enabled: bool = True

    # ---- Retrieval parameters ----
    bm25_top_k: int = 5
    dense_top_k: int = 5
    section_max_chars: int = 3000
    fusion_k: int = 60  # RRF constant

    # ---- Agent switches ----
    single_expert_enabled: bool = True
    cross_expert_enabled: bool = True
    multi_expert_enabled: bool = True

    # ---- LLM ----
    model_name: str = "deepseek-v4"

    # ---- Limits ----
    max_total_tokens: int = 60000
    max_retries: int = 3
    concurrency: int = 500

    def to_dict(self) -> dict:
        d = {}
        for f_name in self.__dataclass_fields__:
            v = getattr(self, f_name)
            if f_name == "prior_subjects":
                d[f_name] = [s["category"] for s in v]
            else:
                d[f_name] = v
        return d
