"""
Main pipeline orchestrator — replaces UsAggreAnalyzer._analyze().

Phase 1: Multi-path retrieval with prior subjects
Phase 2: Single-report analysis (per report)
Phase 3: Cross-report trend analysis (per subject)
Phase 4: Two-step aggregation → final JSON
Phase 5: Evaluation (4-dim LLM judge) — called externally

Supports ablation switches via PipelineConfig.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import aiohttp

from config import PipelineConfig
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.dense_retriever import retrieve_dense
from retrieval.section_retriever import retrieve_sections
from retrieval.fusion import fuse_and_concat
from agents.single_report_agent import run_single_report_analyses
from agents.cross_report_agent import pivot_by_subject, run_cross_report_analyses
from agents.aggregation_agent import AggregationAgent
from agents.base_agent import count_tokens
from verification.retrieval_verifier import verify_retrieval


# ---- EDGAR report reader ----

def _read_report_texts(report_path: Path) -> dict[str, str]:
    """Read all text files from an EDGAR report folder.

    Returns {filename: content} dict.
    """
    texts = {}
    for fname in ["full_report.txt", "item7.txt", "item8.txt", "report.json"]:
        fpath = report_path / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                texts[fname] = f.read()
    return texts


def _build_report_name(report_json: dict) -> str:
    """Build Chinese-style report name from report.json metadata."""
    meta = report_json.get("metadata", {})
    year = meta.get("fiscal_year", "????")
    rtype = meta.get("report_type", "")
    if rtype == "10-K":
        return f"{year}年年度报告 (10-K)"
    elif rtype == "10-Q":
        qtr = meta.get("fiscal_quarter", "")
        qlabel = f"Q{qtr}" if qtr else ""
        return f"{year}年{qlabel}季度报告 (10-Q)"
    return f"{year}年{rtype}报告"


def _format_fraud_period(fraud_period: dict | None) -> str:
    """Format fraud_period dict into human-readable string."""
    if not fraud_period:
        return ""
    sy = fraud_period.get("start_year")
    ey = fraud_period.get("end_year")
    if not sy or not ey:
        return ""
    sq = fraud_period.get("start_quarter", "")
    eq = fraud_period.get("end_quarter", "")
    start_str = f"{sq} {sy}" if sq else str(sy)
    end_str = f"{eq} {ey}" if eq else str(ey)
    return f"{start_str} through {end_str}"


class USPipelineV2:
    """US market financial fraud detection pipeline with prior-guided retrieval."""

    def __init__(self, report_paths: list[Path], config: PipelineConfig | None = None):
        self.report_paths = [Path(p) for p in report_paths]
        self.config = config or PipelineConfig()
        self.aggregator = AggregationAgent(
            model_name=self.config.model_name,
            max_retries=self.config.max_retries,
        )
        self.retrieval_stats: dict = {}
        self.token_stats: dict[str, int] = {}  # per-phase token counts

    async def run(
        self,
        fraud_period: dict | None = None,
        session: aiohttp.ClientSession | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> str:
        """Execute the full pipeline and return final JSON result string.

        Args:
            fraud_period: Optional {start_year, end_year, ...} dict.
            session: Shared aiohttp session.
            sem: Concurrency limiter.

        Returns:
            JSON string with is_risk, risk_facts, market="US".
        """
        fp_str = _format_fraud_period(fraud_period)
        self.token_stats = {}

        # ---- Phase 1: Multi-path retrieval ----
        logging.info("Phase 1: Multi-path retrieval")
        report_data = await self._retrieve_all(session, sem)

        # Track retrieval-side tokens (raw = pre-fusion, fused = post-truncation)
        retrieval_raw = sum(rd.get("retrieval_raw_tokens", 0) for rd in report_data)
        retrieval_fused = sum(
            count_tokens(rd["retrieved_text"]) for rd in report_data
        )
        self.token_stats["retrieval_raw"] = retrieval_raw
        self.token_stats["retrieval_fused"] = retrieval_fused

        # ---- Phase 2: Single-report analysis ----
        if self.config.single_expert_enabled:
            logging.info("Phase 2: Single-report analysis")
            single_results, sr_tokens = await self._single_report_phase(report_data, fp_str, session, sem)
            for phase, tokens in sr_tokens.items():
                self.token_stats[phase] = self.token_stats.get(phase, 0) + tokens
        else:
            single_results = []

        # ---- Phase 3: Cross-report trend analysis ----
        if self.config.cross_expert_enabled:
            logging.info("Phase 3: Cross-report trend analysis")
            cross_results, cr_tokens = await self._cross_report_phase(report_data, fp_str, session, sem)
            for phase, tokens in cr_tokens.items():
                self.token_stats[phase] = self.token_stats.get(phase, 0) + tokens
        else:
            cross_results = []

        # ---- Phase 4: Aggregation ----
        logging.info("Phase 4: Aggregation")
        if self.config.multi_expert_enabled:
            final = await self.aggregator.aggregate(
                single_results=single_results,
                cross_results=cross_results,
                fraud_period_str=fp_str,
                session=session,
                sem=sem,
            )
        else:
            # Ablation: single LLM directly on retrieved text
            all_retrieved = "\n\n".join(
                f"=== {rd['report_name']} ===\n{rd['retrieved_text']}"
                for rd in report_data
            )
            final = await self.aggregator.single_llm_aggregate(
                retrieved_text=all_retrieved,
                fraud_period_str=fp_str,
                session=session,
                sem=sem,
            )

        # Collect token usage from aggregation agent
        for phase, tokens in self.aggregator.token_usage.items():
            self.token_stats[phase] = self.token_stats.get(phase, 0) + tokens

        return final

    async def _retrieve_all(
        self, session, sem
    ) -> list[dict]:
        """Run multi-path retrieval for all reports.

        Returns list of {report_name, report_path, retrieved_text, sections} dicts.
        """
        subjects = self.config.prior_subjects
        logging.info(f"  Subjects: {len(subjects)}, BM25={self.config.bm25_enabled}, "
                     f"Dense={self.config.dense_enabled}, Section={self.config.section_enabled}")

        report_data = []
        for rp in self.report_paths:
            texts = _read_report_texts(rp)
            try:
                report_json = json.loads(texts.get("report.json", "{}"))
            except json.JSONDecodeError:
                report_json = {}
            report_name = _build_report_name(report_json)

            # Use full_report.txt for BM25+dense, item8.txt for section extraction
            full_text = texts.get("full_report.txt", "")
            item8_text = texts.get("item8.txt", "")
            search_text = full_text if full_text else f"{texts.get('item7.txt', '')}\n{item8_text}"

            path_results = []

            if self.config.bm25_enabled and search_text:
                bm25_res = retrieve_bm25(search_text, subjects, top_k=self.config.bm25_top_k)
                path_results.append(bm25_res)

            if self.config.dense_enabled and search_text:
                dense_res = retrieve_dense(search_text, subjects, top_k=self.config.dense_top_k)
                path_results.append(dense_res)

            if self.config.section_enabled and item8_text:
                section_res = retrieve_sections(
                    item8_text, subjects, max_chars=self.config.section_max_chars
                )
                path_results.append(section_res)

            # Fuse and concatenate
            fused_text = fuse_and_concat(
                path_results,
                k=self.config.fusion_k,
                max_tokens=self.config.max_total_tokens // len(self.report_paths),
            )
            if not fused_text and search_text:
                # Fallback: use truncated full text
                from agents.base_agent import truncate_by_tokens
                fused_text = truncate_by_tokens(
                    search_text, self.config.max_total_tokens // len(self.report_paths)
                )

            # Count raw tokens before fusion/truncation
            raw_tokens = 0
            for path_res in path_results:
                for hits in path_res.values():
                    raw_tokens += sum(count_tokens(h["text"]) for h in hits)

            report_data.append({
                "report_name": report_name,
                "report_path": rp,
                "retrieved_text": fused_text,
                "path_results": path_results,
                "retrieval_raw_tokens": raw_tokens,
            })

        # Verification
        self.retrieval_stats = verify_retrieval(report_data, self.config)

        return report_data

    async def _single_report_phase(
        self, report_data: list[dict], fp_str: str, session, sem
    ) -> tuple[list[dict], dict[str, int]]:
        """Run single-report analysis for all reports."""
        tasks = []
        for rd in report_data:
            tasks.append({
                "report_name": rd["report_name"],
                "retrieved_text": rd["retrieved_text"],
            })
        return await run_single_report_analyses(
            report_retrieved=tasks,
            fraud_period_str=fp_str,
            model_name=self.config.model_name,
            session=session,
            sem=sem,
        )

    async def _cross_report_phase(
        self, report_data: list[dict], fp_str: str, session, sem
    ) -> tuple[list[dict], dict[str, int]]:
        """Run cross-report trend analysis by pivoting subjects across reports.

        Skips subjects with < MIN_SUBJECT_CHARS content or appearing in only 1 report.
        """
        MIN_SUBJECT_CHARS = 400  # skip subjects with too little content
        MIN_REPORTS = 2           # skip subjects that only appear in 1 report

        # Build per-report per-subject sections from retrieval results
        report_sections: dict[str, dict[str, str]] = {}
        for rd in report_data:
            subj_texts = {}
            for path_res in rd["path_results"]:
                for cat, hits in path_res.items():
                    combined = "\n...\n".join(h["text"] for h in hits)
                    if cat in subj_texts:
                        subj_texts[cat] += "\n...\n" + combined
                    else:
                        subj_texts[cat] = combined
            if subj_texts:
                report_sections[rd["report_name"]] = subj_texts

        # Pivot to per-subject cross-report content
        subject_content = pivot_by_subject(report_sections)

        # Filter: skip subjects with too little content or only 1 report
        filtered = {}
        skipped = 0
        for subj, content in subject_content.items():
            n_reports = content.count("=== ")
            if len(content) < MIN_SUBJECT_CHARS:
                skipped += 1
                continue
            if n_reports < MIN_REPORTS:
                skipped += 1
                continue
            filtered[subj] = content

        logging.info(
            f"  Cross-report subjects: {len(filtered)} (skipped {skipped} "
            f"with <{MIN_SUBJECT_CHARS} chars or <{MIN_REPORTS} reports)"
        )

        return await run_cross_report_analyses(
            subject_content=filtered,
            fraud_period_str=fp_str,
            model_name=self.config.model_name,
            session=session,
            sem=sem,
        )
