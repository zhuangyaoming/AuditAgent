"""
Entry point for US Pipeline V2 — prior-guided multi-path retrieval + LLM Judge eval.

Usage:
  python us_pipeline_v2/run.py --limit 10                          # single case test
  python us_pipeline_v2/run.py --limit 10 --ablation baseline --output us_pipeline_v2/results/baseline.xlsx

  python us_pipeline_v2/run.py --limit 50 --ablation cn_to_us --output us_pipeline_v2/results/cn_to_us.xlsx
  python us_pipeline_v2/run.py --limit 50 --ablation cn_us_to_us --output us_pipeline_v2/results/cn_us_to_us.xlsx

  python us_pipeline_v2/run.py --limit 10 --ablation no_prior --output us_pipeline_v2/results/no_prior.xlsx
  python us_pipeline_v2/run.py --limit 10 --ablation no_hybrid --output us_pipeline_v2/results/no_hybrid.xlsx
  python us_pipeline_v2/run.py --limit 10 --ablation no_multi_expert --output us_pipeline_v2/results/no_multi_expert.xlsx
  python us_pipeline_v2/run.py --limit 10 --ablation no_cross_doc --output us_pipeline_v2/results/no_cross_doc.xlsx

  python us_pipeline_v2/run.py --reeval-only                       # re-eval existing
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import pandas as pd

# Ensure us_pipeline_v2 package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from us_pipeline_v2.pipeline import USPipelineV2
from us_pipeline_v2.config import PipelineConfig
from us_pipeline_v2.ablation import ABLATION_PRESETS
from us_pipeline_v2.evaluation.llm_judge import (
    LLM_CONCURRENCY,
    build_gt_for_us_case,
    compute_case_metrics,
    convert_gt_to_label_format,
    llm_eval_risk_fact,
)
from us_pipeline_v2.evaluation.metrics_aggregator import (
    compute_macro_micro,
    print_summary_table,
)

# ---- Paths ----
CASES_BASE = Path(r"d:\mainfiles\AuditAgent\finfraud_processing\data\processed\cases")
EDGAR_TEXT_BASE = Path(r"d:\mainfiles\AuditAgent\finfraud_processing\data\edgar_reports_text")
DEFAULT_OUTPUT = Path(r"d:\mainfiles\AuditAgent\us_pipeline_v2\results\auditagent_us_pipeline_v2.xlsx")

# ---- I/O bounds for high concurrency ----
DEFAULT_CASE_CONCURRENCY = 500
PIPELINE_LLM_CONCURRENCY = 250   # deepseek agent API concurrency
DISK_IO_CONCURRENCY = 100        # limit concurrent disk reads
AIOHTTP_POOL_LIMIT = 600         # total connection pool
AIOHTTP_POOL_PER_HOST = 300      # per-host connection limit
FLUSH_INTERVAL_SEC = 30         # auto-flush Excel every N seconds
FLUSH_BATCH_SIZE = 50           # auto-flush Excel every N cases

# ---- Sheet schemas ----
SHEET1_COLUMNS = [
    "Id", "CaseId", "CompanyName", "InputPath", "ReportCount",
    "FraudPeriod", "FraudTypes", "RiskAnalysis", "Config",
]
EVAL_PROCESS_COLUMNS = [
    "Id", "CaseId", "CompanyName", "GT_Raw_Count", "GT_Dedup_Count",
    "GT_Duplicates", "GT_Dedup_Issues",
    "risk1", "risk2", "risk3", "risk4", "risk5",
    "eval_1", "eval_2", "eval_3", "eval_4", "eval_5",
]


class ExcelBatchWriter:
    """High-concurrency Excel writer with in-memory buffering and atomic flushes.

    All mutations operate on in-memory DataFrames protected by an asyncio lock.
    A background flush task periodically writes to a temp file and atomically
    replaces the target, preventing file corruption on crash.
    """

    def __init__(
        self,
        excel_path: Path,
        flush_interval: float = FLUSH_INTERVAL_SEC,
        flush_batch: int = FLUSH_BATCH_SIZE,
    ):
        self._path = Path(excel_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._flush_interval = flush_interval
        self._flush_batch = flush_batch

        # In-memory DataFrames
        self._df_sheet1: pd.DataFrame = pd.DataFrame(columns=SHEET1_COLUMNS)
        self._df_metrics: pd.DataFrame = pd.DataFrame()
        self._df_process: pd.DataFrame = pd.DataFrame(columns=EVAL_PROCESS_COLUMNS)

        # Tracking
        self._pending_count: int = 0
        self._flush_event = asyncio.Event()
        self._flush_task: asyncio.Task | None = None
        self._dirty: bool = False

        # Load existing data if file exists
        self._load_existing()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Load existing sheets from disk into memory."""
        if not self._path.exists():
            return
        try:
            self._df_sheet1 = pd.read_excel(self._path, sheet_name="Sheet1")
        except Exception:
            self._df_sheet1 = pd.DataFrame(columns=SHEET1_COLUMNS)

        try:
            self._df_metrics = pd.read_excel(self._path, sheet_name="Eval_Metrics")
        except Exception:
            self._df_metrics = pd.DataFrame()

        try:
            self._df_process = pd.read_excel(self._path, sheet_name="Eval_Process")
        except Exception:
            self._df_process = pd.DataFrame(columns=EVAL_PROCESS_COLUMNS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_existing_case_ids(self) -> set[str]:
        """Return set of CaseIds already in Sheet1."""
        if self._df_sheet1.empty or "CaseId" not in self._df_sheet1.columns:
            return set()
        return set(self._df_sheet1["CaseId"].astype(str).tolist())

    def get_existing_eval_ids(self) -> set[str]:
        """Return set of CaseIds already in Eval_Metrics."""
        if self._df_metrics.empty or "CaseId" not in self._df_metrics.columns:
            return set()
        return set(self._df_metrics["CaseId"].astype(str).tolist())

    def get_sheet1_records(self) -> list[dict]:
        """Return Sheet1 as list of dicts."""
        if self._df_sheet1.empty:
            return []
        return self._df_sheet1.to_dict("records")

    async def upsert_sheet1(self, case_data: dict) -> int:
        """Insert or update a row in Sheet1. Returns the row Id (sheet1_id)."""
        async with self._lock:
            case_id = case_data["CaseId"]
            mask = self._df_sheet1["CaseId"].astype(str) == str(case_id)
            if mask.any():
                idx = self._df_sheet1[mask].index[0]
                for col in ["CompanyName", "InputPath", "ReportCount",
                            "FraudPeriod", "FraudTypes", "RiskAnalysis", "Config"]:
                    self._df_sheet1.loc[idx, col] = case_data.get(col, "")
                sheet1_id = int(self._df_sheet1.loc[idx, "Id"])
            else:
                sheet1_id = len(self._df_sheet1) + 1
                new_row = {"Id": sheet1_id, **case_data}
                self._df_sheet1 = pd.concat(
                    [self._df_sheet1, pd.DataFrame([new_row])], ignore_index=True
                )
            self._mark_dirty()
        return sheet1_id

    async def upsert_metrics(self, case_id: str, eval_data: dict) -> None:
        """Insert or update a row in Eval_Metrics."""
        async with self._lock:
            row_dict = {"CaseId": case_id}
            row_dict.update({k: v for k, v in eval_data.items() if k != "CaseId"})

            if (not self._df_metrics.empty
                    and "CaseId" in self._df_metrics.columns
                    and str(case_id) in self._df_metrics["CaseId"].astype(str).values):
                idx = self._df_metrics[
                    self._df_metrics["CaseId"].astype(str) == str(case_id)
                ].index[0]
                for k, v in row_dict.items():
                    if k in self._df_metrics.columns:
                        self._df_metrics.loc[idx, k] = v
            else:
                self._df_metrics = pd.concat(
                    [self._df_metrics, pd.DataFrame([row_dict])], ignore_index=True
                )
            self._mark_dirty()

    async def upsert_process(self, process_row: dict) -> None:
        """Insert or update a row in Eval_Process."""
        async with self._lock:
            row_dict = {k: process_row.get(k, "") for k in EVAL_PROCESS_COLUMNS}
            case_id = process_row.get("CaseId", "")

            if (not self._df_process.empty
                    and "CaseId" in self._df_process.columns
                    and str(case_id) in self._df_process["CaseId"].astype(str).values):
                idx = self._df_process[
                    self._df_process["CaseId"].astype(str) == str(case_id)
                ].index[0]
                for k, v in row_dict.items():
                    if k in self._df_process.columns:
                        self._df_process.loc[idx, k] = v
            else:
                self._df_process = pd.concat(
                    [self._df_process, pd.DataFrame([row_dict])], ignore_index=True
                )
            self._mark_dirty()

    async def write_summary(self) -> None:
        """Compute macro/micro averages and append SUMMARY row to Eval_Metrics."""
        async with self._lock:
            df = self._df_metrics[
                self._df_metrics["CaseId"].astype(str) != "SUMMARY"
            ]
            if len(df) == 0:
                print("  No cases in Eval_Metrics, skipping summary")
                return

            agg = compute_macro_micro(df)
            summary = {"CaseId": "SUMMARY", "CompanyName": f"N={agg['N']}"}
            summary.update(agg)

            for col in summary:
                if col not in self._df_metrics.columns:
                    self._df_metrics[col] = None
            self._df_metrics = pd.concat(
                [self._df_metrics, pd.DataFrame([summary])], ignore_index=True
            )
            self._mark_dirty()

            # Force immediate flush for summary
            await self._flush_to_disk()
            print_summary_table(agg)

    # ------------------------------------------------------------------
    # Background flush
    # ------------------------------------------------------------------

    def start_background_flush(self) -> None:
        """Launch the periodic flush coroutine."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop_background_flush(self) -> None:
        """Cancel the background flush loop and do a final flush."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self._flush_to_disk()

    async def _flush_loop(self) -> None:
        """Periodic flush: every N seconds or when signaled."""
        while True:
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(), timeout=self._flush_interval
                )
            except asyncio.TimeoutError:
                pass  # interval elapsed → flush below
            self._flush_event.clear()
            async with self._lock:
                if self._dirty:
                    await self._flush_to_disk()

    def _mark_dirty(self) -> None:
        """Mark data dirty and maybe signal early flush."""
        self._dirty = True
        self._pending_count += 1
        if self._pending_count >= self._flush_batch:
            self._flush_event.set()
            self._pending_count = 0

    async def _flush_to_disk(self) -> None:
        """Write all three sheets to a temp file, then atomically replace target.

        Must be called while holding self._lock.
        """
        if not self._dirty:
            return

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".xlsx", dir=self._path.parent, prefix=".auditagent_"
        )
        os.close(tmp_fd)

        try:
            # Write to temp file (openpyxl engine doesn't support mode="a"
            # for new files, so always create fresh)
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                self._df_sheet1.to_excel(writer, sheet_name="Sheet1", index=False)
                self._df_metrics.to_excel(writer, sheet_name="Eval_Metrics", index=False)
                self._df_process.to_excel(writer, sheet_name="Eval_Process", index=False)

            # Atomic replace
            os.replace(tmp_path, self._path)
            self._dirty = False
            self._pending_count = 0
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _read_gt_file(gt_path: Path, io_sem: asyncio.Semaphore | None = None) -> dict | None:
    """Read ground-truth JSON with optional I/O semaphore (synchronous)."""
    if not gt_path.exists():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_edgar_folders() -> list[dict]:
    """Scan EDGAR_TEXT_BASE, return list of case dicts."""
    cases = []
    for folder in sorted(EDGAR_TEXT_BASE.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        if len(parts) < 2:
            continue
        case_id = parts[0]
        company_name = parts[1].replace("_", " ")
        report_paths = sorted([p for p in folder.iterdir() if p.is_dir()])
        if not report_paths:
            continue
        cases.append({
            "CaseId": case_id,
            "CompanyName": company_name,
            "InputPath": ";".join(str(p) for p in report_paths),
            "ReportCount": len(report_paths),
            "FraudPeriod": "",
            "FraudTypes": "",
            "RiskAnalysis": None,
        })
    return cases


async def process_case(
    case_id: str,
    company: str,
    paths: list[str],
    writer: ExcelBatchWriter,
    session: aiohttp.ClientSession,
    pipeline_sem: asyncio.Semaphore,
    eval_sem: asyncio.Semaphore,
    config: PipelineConfig,
    config_json: str = "",
    io_sem: asyncio.Semaphore | None = None,
) -> dict | None:
    """Run US Pipeline V2 + LLM Judge on a single case."""
    print(f"\n  {case_id}: {company}")
    print(f"    {len(paths)} reports, config: prior={config.prior_enabled} "
          f"bm25={config.bm25_enabled} dense={config.dense_enabled} "
          f"section={config.section_enabled}")

    # Load ground truth (disk read — may be throttled by io_sem via caller)
    gt_path = CASES_BASE / f"case_{case_id}.json"
    gt_case = None
    fp = None
    if gt_path.exists():
        if io_sem:
            async with io_sem:
                gt_case = _read_gt_file(gt_path)
        else:
            gt_case = _read_gt_file(gt_path)
        if gt_case:
            fp = gt_case.get("fraud_period", {})
            if fp:
                print(f"    fraud_period: {fp.get('start_year', '')}-{fp.get('end_year', '')}")

    # Run Pipeline
    result_str = None
    token_stats: dict = {}
    try:
        path_objects = [Path(p) for p in paths]
        pipeline = USPipelineV2(path_objects, config=config)
        result_str = await pipeline.run(
            fraud_period=fp if fp else None,
            session=session,
            sem=pipeline_sem,
        )
        parsed = json.loads(result_str)
        is_risk = parsed.get("is_risk", "?")
        n_facts = len(parsed.get("risk_facts", []))
        token_stats = pipeline.token_stats.copy()
        print(f"    Pipeline: is_risk={is_risk}, risk_facts={n_facts}")
        if pipeline.retrieval_stats:
            stats = pipeline.retrieval_stats
            print(f"    Retrieval: {stats.get('subjects_with_hits', '?')}/{stats.get('n_subjects', '?')} "
                  f"subjects hit, {stats.get('total_passages', '?')} passages")
    except Exception as e:
        print(f"    Pipeline ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Save Agent result to Sheet1 (in-memory, fast)
    sheet1_id = 0
    if result_str:
        agent_data = {
            "CaseId": case_id,
            "CompanyName": company,
            "InputPath": ";".join(paths),
            "ReportCount": len(paths),
            "FraudPeriod": (
                f"{fp.get('start_year', '')}-{fp.get('end_year', '')}"
                if fp else ""
            ),
            "FraudTypes": (
                gt_case.get("fraud_types", [{}])[0].get("type_l2", "")
                if gt_case and gt_case.get("fraud_types") else ""
            ),
            "RiskAnalysis": result_str,
            "Config": config_json,
        }
        sheet1_id = await writer.upsert_sheet1(agent_data)
        print(f"    Agent result saved to Sheet1")

    # LLM Judge evaluation
    if not gt_path.exists() or not gt_case:
        print(f"    SKIP: No ground truth")
        return None

    try:
        pred = json.loads(result_str) if isinstance(result_str, str) else result_str
    except (json.JSONDecodeError, TypeError):
        print(f"    SKIP: JSON parse failed")
        return None

    try:
        gt = build_gt_for_us_case(gt_case)
        gt_label_json = convert_gt_to_label_format(gt)
        n_gt_dedup = len(gt["affected_accounts_dedup"])
        n_gt_evidence = gt["_n_gt_evidence"]
        risk_facts = pred.get("risk_facts", [])
        n_risk_facts = min(len(risk_facts), 5)
        top_is_risk = str(pred.get("is_risk", ""))

        from us_pipeline_v2.agents.base_agent import count_tokens
        judge_tokens = 0

        evals: dict[int, dict] = {}
        risk_cols = {}
        eval_cols = {}

        if n_risk_facts == 0:
            for j in range(5):
                risk_cols[f"risk{j+1}"] = ""
                eval_cols[f"eval_{j+1}"] = ""
            metrics = compute_case_metrics({}, 0, n_gt_dedup, n_gt_evidence)
        else:
            for j in range(5):
                if j < n_risk_facts:
                    rf = risk_facts[j]
                    risk_cols[f"risk{j+1}"] = json.dumps(rf, ensure_ascii=False)
                    eval_result = await llm_eval_risk_fact(
                        session, eval_sem, rf, top_is_risk, gt_label_json,
                    )
                    evals[j] = eval_result
                    eval_cols[f"eval_{j+1}"] = json.dumps(eval_result, ensure_ascii=False)
                    score_str = "/".join(
                        str(eval_result.get(k, 0)) for k in
                        ["is_risk", "risk_title", "involved_report", "evidence_chain"]
                    )
                    print(f"    LLM Judge [{j+1}/{n_risk_facts}]: {score_str}")
                else:
                    risk_cols[f"risk{j+1}"] = ""
                    eval_cols[f"eval_{j+1}"] = ""

            metrics = compute_case_metrics(evals, n_risk_facts, n_gt_dedup, n_gt_evidence)

        # Merge token stats
        token_stats["llm_judge"] = judge_tokens
        token_stats["token_total"] = sum(token_stats.values())
        print(f"    Tokens: {token_stats}")

        print(f"    LLM Judge: R_I={metrics['R_I']:.3f}, P_I={metrics['P_I']:.3f}, "
              f"F1_I={metrics['F1_I']:.3f}, R_E={metrics['R_E']:.3f}")

        # Save Eval_Process (in-memory, fast)
        dedup_issues_json = json.dumps(gt["affected_accounts_dedup"], ensure_ascii=False)
        process_row = {
            "Id": sheet1_id,
            "CaseId": case_id,
            "CompanyName": company,
            "GT_Raw_Count": n_gt_dedup,
            "GT_Dedup_Count": n_gt_dedup,
            "GT_Duplicates": "",
            "GT_Dedup_Issues": dedup_issues_json,
            **risk_cols,
            **eval_cols,
        }
        await writer.upsert_process(process_row)

        # Save Eval_Metrics (in-memory, fast)
        eval_data = {**metrics, "CompanyName": company}
        for tk, tv in token_stats.items():
            eval_data[f"Tokens_{tk}"] = tv
        await writer.upsert_metrics(case_id, eval_data)
        print(f"    LLM Judge result saved")

        return {"CaseId": case_id, "CompanyName": company, **metrics}
    except Exception as e:
        print(f"    LLM Judge ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def reeval_case(
    case_data: dict,
    writer: ExcelBatchWriter,
    session: aiohttp.ClientSession,
    eval_sem: asyncio.Semaphore,
    config: PipelineConfig,
    io_sem: asyncio.Semaphore | None = None,
) -> dict | None:
    """Re-evaluate an existing case (skip pipeline, only run LLM Judge)."""
    case_id = case_data["CaseId"]
    company = case_data["CompanyName"]
    sheet1_id = int(case_data.get("Id", 0))

    print(f"\n  {case_id}: {company}")

    gt_path = CASES_BASE / f"case_{case_id}.json"
    if not gt_path.exists():
        print(f"    SKIP: GT not found")
        return None

    if io_sem:
        async with io_sem:
            gt_case = json.loads(gt_path.read_text(encoding="utf-8"))
    else:
        gt_case = json.loads(gt_path.read_text(encoding="utf-8"))

    risk_analysis = case_data.get("RiskAnalysis", "")
    if not risk_analysis:
        print(f"    SKIP: No RiskAnalysis data")
        return None

    try:
        pred = json.loads(risk_analysis) if isinstance(risk_analysis, str) else risk_analysis
    except json.JSONDecodeError:
        print(f"    SKIP: Failed to parse RiskAnalysis")
        return None

    try:
        gt = build_gt_for_us_case(gt_case)
        gt_label_json = convert_gt_to_label_format(gt)
        n_gt_dedup = len(gt["affected_accounts_dedup"])
        n_gt_evidence = gt["_n_gt_evidence"]
        risk_facts = pred.get("risk_facts", [])
        n_risk_facts = min(len(risk_facts), 5)
        top_is_risk = str(pred.get("is_risk", ""))

        from us_pipeline_v2.agents.base_agent import count_tokens
        judge_tokens = 0
        token_stats = {}

        evals: dict[int, dict] = {}
        risk_cols = {}
        eval_cols = {}

        if n_risk_facts == 0:
            for j in range(5):
                risk_cols[f"risk{j+1}"] = ""
                eval_cols[f"eval_{j+1}"] = ""
            metrics = compute_case_metrics({}, 0, n_gt_dedup, n_gt_evidence)
        else:
            for j in range(5):
                if j < n_risk_facts:
                    rf = risk_facts[j]
                    risk_cols[f"risk{j+1}"] = json.dumps(rf, ensure_ascii=False)
                    eval_result = await llm_eval_risk_fact(
                        session, eval_sem, rf, top_is_risk, gt_label_json,
                    )
                    evals[j] = eval_result
                    eval_cols[f"eval_{j+1}"] = json.dumps(eval_result, ensure_ascii=False)
                    score_str = "/".join(
                        str(eval_result.get(k, 0)) for k in
                        ["is_risk", "risk_title", "involved_report", "evidence_chain"]
                    )
                    print(f"    LLM Judge [{j+1}/{n_risk_facts}]: {score_str}")
                else:
                    risk_cols[f"risk{j+1}"] = ""
                    eval_cols[f"eval_{j+1}"] = ""

            metrics = compute_case_metrics(evals, n_risk_facts, n_gt_dedup, n_gt_evidence)

        token_stats["llm_judge"] = judge_tokens
        token_stats["token_total"] = sum(token_stats.values())
        print(f"    Tokens: {token_stats}")

        print(f"    LLM Judge: R_I={metrics['R_I']:.3f}, P_I={metrics['P_I']:.3f}, "
              f"F1_I={metrics['F1_I']:.3f}, R_E={metrics['R_E']:.3f}")

        process_row = {
            "Id": sheet1_id,
            "CaseId": case_id,
            "CompanyName": company,
            "GT_Raw_Count": n_gt_dedup,
            "GT_Dedup_Count": n_gt_dedup,
            "GT_Duplicates": "",
            "GT_Dedup_Issues": json.dumps(gt["affected_accounts_dedup"], ensure_ascii=False),
            **risk_cols,
            **eval_cols,
        }
        await writer.upsert_process(process_row)

        eval_data = {**metrics, "CompanyName": company}
        if token_stats:
            for tk, tv in token_stats.items():
                eval_data[f"Tokens_{tk}"] = tv
        await writer.upsert_metrics(case_id, eval_data)
        print(f"    LLM Judge result saved")

        return {"CaseId": case_id, "CompanyName": company, **metrics}
    except Exception as e:
        print(f"    LLM Judge ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="US Pipeline V2 — prior-guided multi-path retrieval + LLM Judge eval"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of cases to process (0 = no limit)")
    parser.add_argument("--ablation", type=str, default="baseline",
                        choices=list(ABLATION_PRESETS.keys()),
                        help="Ablation preset to use")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output Excel file path")
    parser.add_argument("--reeval-only", action="store_true",
                        help="Only run LLM Judge on existing cases, skip pipeline")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CASE_CONCURRENCY,
                        help=f"Case-level concurrency (default {DEFAULT_CASE_CONCURRENCY})")
    parser.add_argument("--io-concurrency", type=int, default=DISK_IO_CONCURRENCY,
                        help=f"Max concurrent disk reads (default {DISK_IO_CONCURRENCY})")
    parser.add_argument("--flush-interval", type=float, default=FLUSH_INTERVAL_SEC,
                        help=f"Excel auto-flush interval in seconds (default {FLUSH_INTERVAL_SEC})")
    parser.add_argument("--flush-batch", type=int, default=FLUSH_BATCH_SIZE,
                        help=f"Excel auto-flush every N cases (default {FLUSH_BATCH_SIZE})")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Build config from ablation preset
    config = ABLATION_PRESETS[args.ablation]()
    config.concurrency = args.concurrency
    config_json = json.dumps(config.to_dict(), ensure_ascii=False)
    output_base = Path(args.output)
    output_path = output_base.parent / f"{output_base.stem}_{config.model_name}{output_base.suffix}"

    print("=" * 80)
    print(f"US Pipeline V2 — Ablation: {args.ablation}")
    print(f"  prior_enabled={config.prior_enabled} ({len(config.prior_subjects)} subjects)")
    print(f"  bm25={config.bm25_enabled} dense={config.dense_enabled} section={config.section_enabled}")
    print(f"  single_expert={config.single_expert_enabled} cross_expert={config.cross_expert_enabled}")
    print(f"  multi_expert={config.multi_expert_enabled}")
    print(f"  concurrency={args.concurrency}  io_concurrency={args.io_concurrency}")
    print(f"  output={output_path}")
    print("=" * 80)

    # ---- Excel batch writer (in-memory + atomic flushes) ----
    writer = ExcelBatchWriter(
        output_path,
        flush_interval=args.flush_interval,
        flush_batch=args.flush_batch,
    )
    writer.start_background_flush()

    # ---- aiohttp connector with high pool limits ----
    connector = aiohttp.TCPConnector(
        limit=AIOHTTP_POOL_LIMIT,
        limit_per_host=AIOHTTP_POOL_PER_HOST,
        ttl_dns_cache=300,
    )

    # ---- Semaphores ----
    case_sem = asyncio.Semaphore(args.concurrency)
    pipeline_sem = asyncio.Semaphore(PIPELINE_LLM_CONCURRENCY)
    eval_sem = asyncio.Semaphore(LLM_CONCURRENCY)
    io_sem = asyncio.Semaphore(args.io_concurrency)

    try:
        if args.reeval_only:
            # ---- Re-eval mode ----
            if not output_path.exists():
                print(f"ERROR: {output_path} does not exist")
                return

            existing_cases = writer.get_sheet1_records()
            print(f"\nFound {len(existing_cases)} cases in Sheet1")

            existing_eval_ids = writer.get_existing_eval_ids()
            print(f"Found {len(existing_eval_ids)} cases in Eval_Metrics (will skip)")

            to_reeval = [c for c in existing_cases if str(c["CaseId"]) not in existing_eval_ids]
            print(f"Cases to re-evaluate: {len(to_reeval)}")

            if args.limit > 0:
                to_reeval = to_reeval[:args.limit]

            if not to_reeval:
                print("No cases to re-evaluate!")
                return

            async with aiohttp.ClientSession(connector=connector) as session:
                async def run_one(case, idx):
                    async with case_sem:
                        print(f"\n[{idx + 1}/{len(to_reeval)}] Processing {case['CaseId']}")
                        return await reeval_case(
                            case, writer, session, eval_sem, config, io_sem,
                        )

                tasks = [asyncio.create_task(run_one(c, i)) for i, c in enumerate(to_reeval)]
                await asyncio.gather(*tasks)

            await writer.write_summary()
            return

        # ---- Normal mode: scan + process ----
        all_cases = scan_edgar_folders()
        print(f"Total folders in edgar_reports_text: {len(all_cases)}")

        existing_case_ids = writer.get_existing_case_ids()
        if existing_case_ids:
            print(f"Already processed in this output: {len(existing_case_ids)} cases")

        unprocessed = [c for c in all_cases if c["CaseId"] not in existing_case_ids]
        total_new = len(unprocessed)
        if args.limit > 0:
            unprocessed = unprocessed[:args.limit]
            print(f"Limit: processing {len(unprocessed)} of {total_new} new cases")
        else:
            print(f"New cases to process: {len(unprocessed)}")

        if not unprocessed:
            print("Nothing to do!")
            return

        async with aiohttp.ClientSession(connector=connector) as session:
            async def run_one(case, idx):
                async with case_sem:
                    print(f"\n[{idx + 1}/{len(unprocessed)}] Processing {case['CaseId']}")
                    paths = [p.strip() for p in str(case["InputPath"]).split(";") if p.strip()]
                    return await process_case(
                        case["CaseId"], case["CompanyName"], paths,
                        writer, session, pipeline_sem, eval_sem, config, config_json,
                        io_sem,
                    )

            tasks = [asyncio.create_task(run_one(c, i)) for i, c in enumerate(unprocessed)]
            results = await asyncio.gather(*tasks)

        processed = sum(1 for r in results if r)
        print(f"\nPipeline complete: {processed}/{len(unprocessed)} cases processed")

        await writer.write_summary()
        print(f"\nResults saved to {output_path}")

    finally:
        await writer.stop_background_flush()


if __name__ == "__main__":
    asyncio.run(main())
