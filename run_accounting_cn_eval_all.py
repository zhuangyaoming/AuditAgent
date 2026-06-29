"""
Run Chinese Hierarchical RAG Agent + metrics_new Evaluator on Chinese financial fraud data.

Uses the original Chinese agent architecture (SingleAnalyzer + CrossAnalyzer + Aggregation)
with pre-parsed Context .txt files (bypassing PDF parsing since most PDFs don't exist locally).
Evaluates with metrics_new.Evaluator only.

Mirrors run_accounting_eval_all.py but adapted for Chinese data.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import aiohttp
import pandas as pd
import tiktoken

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

sys.path.insert(0, str(Path(__file__).parent / "LLM-FinRisk" / "LLM-FinRisk" / "solution" / "Hierarchical_RAG"))
sys.path.insert(0, str(Path(__file__).parent / "LLM-FinRisk" / "LLM-FinRisk" / "evaluation"))

from SingleReportAnalyzer import SingleAnalyzer
from CrossReportAnalyzer import CrossAnalyzer
from prompt import construct_aggre_prompt, construct_cross_report_prompt
from llm_judge import (
    LLM_CONCURRENCY,
    build_gt_for_case,
    compute_case_metrics,
    convert_gt_to_label_format,
    get_dedup_stats,
    llm_eval_risk_fact,
)

# ============================================================
# Constants
# ============================================================
ENCODING_NAME = "cl100k_base"
SEPARATOR = "\n next \n"
CONCURRENCY_LIMIT = 1
MAX_RETRIES = 5
DEFAULT_SYSTEM_ROLE = "你是一个财务审计专家。"

# ============================================================
# Failure handling: distinguish "API failed" from "model said empty"
# ============================================================
# Two failure classes so a flaky/exhausted API never gets silently saved as a
# successful (empty) result:
#   - TerminalAPIError: non-retryable (no balance / quota / auth). Triggers a
#     global stop so we don't keep burning failed calls across remaining cases.
#   - CallFailedError: transient failure that survived all retries. The case is
#     NOT saved, so the next run picks it up again instead of skipping it.

class TerminalAPIError(Exception):
    """Non-retryable API error (insufficient balance / quota / auth)."""


class CallFailedError(Exception):
    """LLM call failed after exhausting retries (transient)."""


# Global circuit breaker. Created in main(); set when a terminal error is seen.
STOP_EVENT: "asyncio.Event | None" = None

# Substrings that indicate a terminal billing/quota/auth problem (case-insensitive;
# Chinese terms are matched as-is).
_TERMINAL_ERROR_KEYWORDS = (
    "insufficient balance", "insufficient_quota", "insufficient quota",
    "exceeded your current quota", "account balance", "billing", "arrears",
    "余额不足", "余额", "配额", "欠费", "账户", "已用尽", "quota exceeded",
)


def _is_terminal_error(status: int, body: str) -> bool:
    """Return True if an HTTP status / response body signals a non-retryable
    billing / quota / auth error."""
    if status in (401, 402, 403):
        return True
    low = (body or "").lower()
    return any(kw in low for kw in _TERMINAL_ERROR_KEYWORDS)


DATASET_XLSX = Path(r"d:\mainfiles\AuditAgent\LLM-FinRisk\LLM-FinRisk\data\dataset\FinFraud-dataset-txt-cross.xlsx")
TXT_PREFIX = Path(r"d:\mainfiles\AuditAgent\llm_risk_backup\llm-financial-risk\dataset\financial-report-context-txt")
REPORT_TXT_BASE = Path(r"d:\mainfiles\AuditAgent\llm_risk_backup\llm-financial-risk\processed_dataset_txt\financial-report")
AUDIT_RESULT_PATH = Path(r"d:\mainfiles\AuditAgent\LLM-FinRisk\LLM-FinRisk\results\auditagent_cn_all_cases.xlsx")

# TODO: Set your API keys via environment variables before running.
# Original keys backed up in api_keys_backup.txt (EXCLUDED from git).
API_CONFIGS = {
    "minimax-m2.5": {
        "url": "https://api.minimax.chat/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "minimax-m2.7-volces": {
        "url": "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "deepseek-v4": {
        "url": "https://api.deepseek.com/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "deepseek-reasoner": {
        "url": "https://api.deepseek.com/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "gpt-4o": {
        "url": "https://api.vectorengine.ai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "o3-mini": {
        "url": "https://api.vectorengine.ai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
}

MODEL_CONFIGS = {
    "minimax-m2.5": {"model": "MiniMax-M2.5"},
    "minimax-m2.7-volces": {"model": "minimax-m2.7"},
    "deepseek-v4": {"model": "deepseek-v4-pro"},
    "deepseek-reasoner": {"model": "deepseek-reasoner"},
    "gpt-4o": {"model": "gpt-4o"},
    "o3-mini": {"model": "o3-mini"},
}

# Context .txt section header pattern: "YYYY年ReportType--TermName"
CONTEXT_SECTION_RE = re.compile(r'^(\d{4}年[^\-\n]+)--(.+)$', re.MULTILINE)

# ---- Prior subject sets for cross-market transfer experiments ----

CN_TERM_LIST = [
    "存货", "应收账款", "货币资金", "其他应收款", "商誉", "固定资产",
    "预付款项", "长期股权投资", "在建工程", "未分配利润",
    "其他非流动资产", "无形资产", "营业收入和营业成本", "财务费用", "其他应付款",
]

# US_PRIOR_15 categories → Chinese keywords for matching CN Context .txt section headers
US_PRIOR_CN_KEYWORDS: dict[str, list[str]] = {
    "Revenue":                      ["营业收入", "收入"],
    "Accounts Receivable":          ["应收账款", "应收票据", "应收款项", "应收"],
    "Net Income":                   ["净利润", "净收益", "净亏损", "净利润"],
    "Additional Paid-in Capital":   ["资本公积", "溢价"],
    "Investment Securities":        ["交易性金融资产", "可供出售金融资产", "持有至到期投资", "债权投资", "其他债权投资", "其他权益工具投资"],
    "Stockholders' Equity":         ["股东权益", "所有者权益", "归属于母公司所有者权益"],
    "Common Stock":                 ["股本", "实收资本", "普通股"],
    "Inventory":                    ["存货", "库存商品", "发出商品"],
    "Operating Expenses":           ["销售费用", "管理费用", "研发费用", "税金及附加"],
    "Compensation Expense":         ["应付职工薪酬", "职工薪酬", "股份支付", "职工"],
    "Cost of Goods Sold":           ["营业成本", "销售成本"],
    "Pre-tax Income":               ["利润总额", "税前利润", "应纳税所得"],
    "Loss Reserves":                ["坏账准备", "减值准备", "资产减值", "信用减值", "跌价准备"],
    "Cash and Cash Equivalents":    ["货币资金", "库存现金", "银行存款"],
    "Property, Plant and Equipment": ["固定资产", "固定", "在建工程"],
}

def _get_cn_keywords_for_prior(prior: str) -> set[str]:
    """Return set of Chinese keywords for section matching based on prior source.

    Args:
        prior: "cn" (default, no filter), "us" (US 15 mapped to CN), "cn_us" (merged CN+US)
    """
    if prior == "us":
        keywords: set[str] = set()
        for kws in US_PRIOR_CN_KEYWORDS.values():
            keywords.update(kws)
        return keywords
    elif prior == "cn_us":
        keywords: set[str] = set(CN_TERM_LIST)
        for kws in US_PRIOR_CN_KEYWORDS.values():
            keywords.update(kws)
        return keywords
    return set()  # "cn" — no filter, use all sections


def _filter_sections_list(sections_list: list, cn_keywords: set[str]) -> list:
    """Filter sections_list to only keep sections matching cn_keywords.

    Args:
        sections_list: [ [(term_name, content), ...], ... ] per report
        cn_keywords: set of Chinese keywords to match against term names

    Returns:
        Filtered sections_list with same structure, empty sections removed
    """
    if not cn_keywords:
        return sections_list
    filtered = []
    for report_sections in sections_list:
        kept = [(term, content) for term, content in report_sections
                if any(kw in term for kw in cn_keywords)]
        # Keep report even if empty (preserves index alignment with content_list)
        filtered.append(kept)
    return filtered


# Regex for numbered section headers in raw financial report .txt files
# e.g. "1、货币资金", "14、资产减值准备明细"
RAW_SECTION_RE = re.compile(r'^(\d+)、(.+)$', re.MULTILINE)


def _parse_raw_report(filepath: Path) -> tuple[str, str, list[tuple[str, str]]]:
    """Parse a raw financial report .txt into (report_name, full_content, sections).

    Sections are extracted by splitting on numbered headers like '1、货币资金'.
    Between two headers is the content for that section.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    fname = filepath.stem  # e.g. "2011年年度报告-70-97"

    # Extract report name: strip trailing "-页码-页码" or "-None-None"
    report_name = fname
    for sep in ["-", "_"]:
        parts = report_name.rsplit(sep, 2)
        if len(parts) == 3 and all(p.replace("-", "").replace("None", "").isdigit() or p == "None"
                                   for p in parts[1:]):
            report_name = parts[0]
            break

    # Find all section headers
    matches = list(RAW_SECTION_RE.finditer(text))
    if not matches:
        return report_name, text, []

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        term_name = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(content) >= 20:  # skip degenerate sections
            sections.append((term_name, content))

    return report_name, text, sections


def _build_inputs_from_raw(symbol: str, cn_keywords: set[str]) -> tuple[
    list[str], list[list[tuple[str, str]]], list[str]
]:
    """Read raw report .txt files for a stock Symbol, extract & filter sections.

    Args:
        symbol: Stock code (e.g. "4" → folder "000004").
        cn_keywords: Chinese keywords for section filtering (empty = no filter).

    Returns:
        (content_list, sections_list, report_names) — same format as _parse_context_txt.
    """
    folder = REPORT_TXT_BASE / symbol.zfill(6)
    if not folder.exists():
        return [], [], []

    report_files = sorted(folder.glob("*.txt"))
    content_list: list[str] = []
    sections_list: list[list[tuple[str, str]]] = []
    report_names: list[str] = []

    for fpath in report_files:
        report_name, full_text, all_sections = _parse_raw_report(fpath)
        content_list.append(full_text)
        report_names.append(report_name)

        if cn_keywords:
            kept = [(term, content) for term, content in all_sections
                    if any(kw in term for kw in cn_keywords)]
        else:
            kept = all_sections
        sections_list.append(kept)

    return content_list, sections_list, report_names


# ============================================================
# Label building (from single_llm_baseline_cn.py)
# ============================================================

def _split_multi_value(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [s.strip() for s in str(value).split(SEPARATOR) if s.strip()]


def _build_cn_affected_accounts(terms_str: str) -> list[dict]:
    terms = _split_multi_value(terms_str)
    seen: set[str] = set()
    result: list[dict] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            result.append({"account": t, "gaap_codification": ""})
    return result


def _build_cn_fraud_locations(
    reports_str: str, terms_str: str, summaries_str: str, evidences_str: str,
) -> list[dict]:
    reports = _split_multi_value(reports_str)
    terms = _split_multi_value(terms_str)
    summaries = _split_multi_value(summaries_str)
    evidences = _split_multi_value(evidences_str)

    n = max(len(reports), len(terms), len(summaries), len(evidences))
    locs: list[dict] = []
    for i in range(n):
        rpt = reports[i] if i < len(reports) else ""
        trm = terms[i] if i < len(terms) else ""
        smy = summaries[i] if i < len(summaries) else ""
        evd = evidences[i] if i < len(evidences) else ""
        enriched = f"[{trm}] {smy}" if trm and smy else (smy or trm)
        locs.append({"report": rpt, "summary": enriched, "evidence": evd})
    return locs


def _row_to_unified_label(row: pd.Series) -> dict:
    affected_accounts = _build_cn_affected_accounts(str(row.get("Term", "")))
    fraud_locations = _build_cn_fraud_locations(
        reports_str=str(row.get("Report", "")),
        terms_str=str(row.get("Term", "")),
        summaries_str=str(row.get("Summary", "")),
        evidences_str=str(row.get("Evidence", "")),
    )
    return {
        "case_id": str(row.get("Id", "")),
        "is_fraud": int(row.get("IsFraud", 0)),
        "fraud_info": _split_multi_value(row.get("FraudInfo", "")),
        "fraud_locations": fraud_locations,
        "affected_accounts": affected_accounts,
    }


# ============================================================
# Context .txt parser
# ============================================================

def _truncate_by_tokens(content_str: str, max_tokens: int = 60000) -> str:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(content_str)
    if len(tokens) <= max_tokens:
        return content_str
    return encoding.decode(tokens[:max_tokens])


def count_tokens(text: str) -> int:
    """Return token count for a text string using cl100k_base encoding."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


def _parse_context_txt(txt_path: Path) -> tuple[list[str], list[list[tuple[str, str]]], list[str]]:
    """Parse a Context .txt file into the structures expected by the analysis pipeline.

    Returns:
        content_list: Per-report full text (one str per report, concatenated from all its term sections)
        sections_list: Per-report list of (term_name, term_content) tuples
        report_names: Per-report names (e.g. "2010年年度报告")
    """
    raw_text = txt_path.read_text(encoding="utf-8")

    # Split by section headers: "YYYY年ReportType--TermName"
    sections_data: list[tuple[str, str, str]] = []  # [(report_name, term_name, content)]
    matches = list(CONTEXT_SECTION_RE.finditer(raw_text))

    for i, match in enumerate(matches):
        report_name = match.group(1).strip()
        term_name = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        content = raw_text[start:end].strip()
        sections_data.append((report_name, term_name, content))

    if not sections_data:
        return [], [], []

    # Deduplicate and preserve order of report names
    seen_reports: set[str] = set()
    report_order: list[str] = []
    for rn, _, _ in sections_data:
        if rn not in seen_reports:
            seen_reports.add(rn)
            report_order.append(rn)

    # Group by report
    report_sections: dict[str, list[tuple[str, str]]] = {rn: [] for rn in report_order}
    report_content_parts: dict[str, list[str]] = {rn: [] for rn in report_order}

    for report_name, term_name, content in sections_data:
        formatted = f"{report_name}--{term_name}\n{content}"
        report_content_parts[report_name].append(formatted)
        report_sections[report_name].append((term_name, content))

    content_list = ["\n\n".join(report_content_parts[rn]) for rn in report_order]
    sections_list = [report_sections[rn] for rn in report_order]

    return content_list, sections_list, report_order


# ============================================================
# Agent analysis pipeline (adapted from AggreAnalyzer._analyze)
# ============================================================

class CnAgentRunner:
    """Runs the Chinese Hierarchical RAG analysis using pre-parsed Context .txt data."""

    def __init__(self, model_name: str = "minimax-m2.5"):
        self.model_name = model_name
        self.session: aiohttp.ClientSession | None = None
        self.token_usage: dict[str, int] = {}  # {phase: total_tokens}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _async_chat(self, prompt: str, name: str = "", flag: str = "",
                          track_phase: str = "") -> str:
        """Async API call with retry logic.

        Raises:
            TerminalAPIError: non-retryable billing/quota/auth error. Also trips
                the global STOP_EVENT so the remaining cases stop quickly.
            CallFailedError: transient error that survived all retries.

        A returned "" now means only that the model genuinely produced empty
        content — never that the call failed — so callers can safely persist
        real results without poisoning resume.
        """
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            raise TerminalAPIError("global stop already triggered by an earlier terminal error")

        config = API_CONFIGS[self.model_name]
        model = MODEL_CONFIGS[self.model_name]["model"]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_ROLE},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "top_p": 0.9,
        }

        last_err = ""
        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.post(
                    config["url"],
                    headers=config["headers"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if _is_terminal_error(resp.status, text):
                            if STOP_EVENT is not None:
                                STOP_EVENT.set()
                            raise TerminalAPIError(f"HTTP {resp.status}: {text[:200]}")
                        last_err = f"HTTP {resp.status}: {text[:200]}"
                        logging.warning(f"{last_err} (attempt {attempt+1}/{MAX_RETRIES})")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        raise CallFailedError(last_err)

                    data = await resp.json()

                    # Some providers return HTTP 200 with an error payload.
                    if isinstance(data, dict) and data.get("error"):
                        err = json.dumps(data["error"], ensure_ascii=False)
                        if _is_terminal_error(200, err):
                            if STOP_EVENT is not None:
                                STOP_EVENT.set()
                            raise TerminalAPIError(err[:200])
                        last_err = err[:200]
                        logging.warning(f"API error payload (attempt {attempt+1}/{MAX_RETRIES}): {last_err}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        raise CallFailedError(last_err)

                    # Track actual API token usage
                    if track_phase:
                        usage = data.get("usage", {})
                        actual_tokens = usage.get("total_tokens", 0)
                        if actual_tokens:
                            self.token_usage[track_phase] = (
                                self.token_usage.get(track_phase, 0) + actual_tokens
                            )

                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    # Strip thinking tags if present
                    think_end = content.find("</think>")
                    if think_end != -1:
                        content = content[think_end + 8:].strip()
                    return content
            except (TerminalAPIError, CallFailedError):
                raise
            except Exception as e:
                last_err = str(e)
                logging.warning(f"API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise CallFailedError(last_err)
        raise CallFailedError(last_err or "unknown error")

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Robustly extract a JSON object from LLM output.

        Handles: thinking tags, markdown code blocks, surrounding prose,
        trailing garbage (extra braces/brackets after the root object).
        Returns the cleaned JSON string or None.
        """
        if not text or not text.strip():
            return None

        t = text.strip()

        # Strip thinking tags
        think_end = t.find("</think>")
        if think_end != -1:
            t = t[think_end + 8:].strip()

        # Strip markdown code blocks
        t = t.replace("```json", "").replace("```", "").strip()

        if not t:
            return None

        # Find JSON object boundaries: first '{' to matching '}'
        start = t.find("{")
        if start == -1:
            return None
        # Walk braces to find matching close
        depth = 0
        end = -1
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None

        candidate = t[start:end + 1]

        # Try raw_decode: handles trailing garbage after the root object
        import json as _json
        decoder = _json.JSONDecoder()
        try:
            obj, idx = decoder.raw_decode(candidate)
            return _json.dumps(obj, ensure_ascii=False)
        except _json.JSONDecodeError:
            pass

        # If raw_decode fails, try trimming trailing characters
        # (handles extra } or ] added by LLM)
        for trim in range(1, min(10, len(candidate))):
            trimmed = candidate[:-trim]
            try:
                obj = _json.loads(trimmed)
                return _json.dumps(obj, ensure_ascii=False)
            except _json.JSONDecodeError:
                continue

        return None

    def _parse_response(self, response: str) -> str:
        """Clean API response. Returns raw cleaned text (extraction happens at aggregation)."""
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return cleaned

    async def analyze(self, txt_path: Path, prior: str = "cn",
                      symbol: str = "") -> tuple[str, int, dict[str, int]]:
        """Run full Hierarchical RAG analysis.

        Args:
            txt_path: Path to Context .txt file (used when prior="cn").
            prior: "cn" (default), "us" (US prior→CN sections), "cn_us" (merged CN+US→CN).
            symbol: Stock code for raw report lookup (used when prior!="cn").

        Returns (result_json, report_count, token_stats)."""
        self.token_usage = {}  # reset per-case

        cn_keywords = _get_cn_keywords_for_prior(prior)

        if prior != "cn" and symbol:
            # Use raw financial report .txt files → extract & filter sections by prior
            content_list, sections_list, report_names = _build_inputs_from_raw(symbol, cn_keywords)
            if not content_list:
                logging.warning(f"No raw reports found for symbol={symbol}")
                return json.dumps({"is_risk": "0", "risk_facts": []}, ensure_ascii=False), 0, {}
            print(f"    Raw reports: {len(content_list)} files, "
                  f"{sum(len(s) for s in sections_list)} filtered sections "
                  f"across {sum(1 for s in sections_list if s)} reports "
                  f"(prior='{prior}', {len(cn_keywords)} keywords)")
        else:
            # Original path: pre-built Context .txt
            content_list, sections_list, report_names = _parse_context_txt(txt_path)
            if cn_keywords:
                sections_list = _filter_sections_list(sections_list, cn_keywords)
                print(f"    Prior filter '{prior}': {len(cn_keywords)} keywords → "
                      f"{sum(len(s) for s in sections_list)} sections across "
                      f"{sum(1 for s in sections_list if s)} reports")

        report_count = len(report_names)

        if not content_list:
            logging.warning(f"No content parsed from {txt_path}")
            return json.dumps({"is_risk": "0", "risk_facts": []}, ensure_ascii=False), 0, {}

        print(f"    Parsed {report_count} reports: {[str(rn) for rn in report_names]}")

        # Use Path objects for report names (SingleAnalyzer & CrossAnalyzer expect these)
        report_paths = [Path(rn) for rn in report_names]

        # Create analyzers
        single_analyzer = SingleAnalyzer(report_paths=report_paths, model_name=self.model_name)
        cross_analyzer = CrossAnalyzer(report_paths=report_paths, model_name=self.model_name)

        # Get task prompts (is_analyze=False returns tasks without running)
        single_tasks = single_analyzer._analyze(content_list=content_list, is_analyze=False)
        cross_tasks = cross_analyzer._analyze(sections_list=sections_list, is_analyze=False)
        all_tasks = single_tasks + cross_tasks

        # Estimate input tokens per phase (pre-truncation)
        sr_input_est = sum(count_tokens(t[2]) for t in single_tasks)
        cr_input_est = sum(count_tokens(t[2]) for t in cross_tasks)
        logging.info(f"  Agent: {len(single_tasks)} single ({sr_input_est} est tokens) + "
                     f"{len(cross_tasks)} cross ({cr_input_est} est tokens) = "
                     f"{len(all_tasks)} tasks")

        # Run all tasks concurrently
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def process_task(task):
            async with sem:
                flag, name, prompt = task
                prompt = _truncate_by_tokens(prompt, 60000)
                # Track phase: single_report or cross_report
                phase = "single_report" if flag != "cross" else "cross_report"
                output = await self._async_chat(prompt, name, flag, track_phase=phase)
                return flag, name, self._parse_response(output)

        tasks_futures = [asyncio.create_task(process_task(t)) for t in all_tasks]
        results = []
        # Any unrecovered failure (terminal or transient) aborts the whole case so
        # it is NOT saved as a half-empty success. Cancel the still-running siblings
        # first to avoid wasting their tokens, then re-raise for process_case().
        try:
            for future in asyncio.as_completed(tasks_futures):
                result = await future
                results.append(result)
                print(f"    [{len(results)}/{len(all_tasks)}] {result[0]}:{result[1][:40] if result[1] else '?'}")
                logging.info(f"处理进度 {len(results)}/{len(all_tasks)}")
        except BaseException:
            for f in tasks_futures:
                if not f.done():
                    f.cancel()
            raise

        # Merge results
        merge_trend_content = "\n\n".join(
            f"{name}--\n{output}"
            for flag, name, output in results
            if flag == "cross" and output
        )
        merge_single_content = "\n\n".join(
            f"{name}--\n{output}"
            for flag, name, output in results
            if flag != "cross" and output
        )

        # Cross-report summary
        print(f"    [{len(all_tasks)+1}/{len(all_tasks)+2}] cross_report_summary")
        cross_report_prompt = construct_cross_report_prompt(merge_trend_content)
        cs_input_est = count_tokens(cross_report_prompt)
        cross_report_prompt = _truncate_by_tokens(cross_report_prompt, 60000)
        cross_output = await self._async_chat(cross_report_prompt, "", "cross",
                                              track_phase="cross_synthesis")

        # Final aggregation
        print(f"    [{len(all_tasks)+2}/{len(all_tasks)+2}] final_aggregation")
        aggre_prompt = construct_aggre_prompt(merge_single_content, cross_output)
        agg_input_est = count_tokens(aggre_prompt)
        aggre_prompt = _truncate_by_tokens(aggre_prompt, 60000)
        aggre_output = await self._async_chat(aggre_prompt, "", "aggre",
                                              track_phase="aggregation")
        aggre_raw = aggre_output  # keep original for fallback

        # Build token stats (mirrors us_pipeline_v2 format)
        token_stats: dict[str, int] = dict(self.token_usage)
        token_stats["single_report_input_est"] = sr_input_est
        token_stats["cross_report_input_est"] = cr_input_est
        token_stats["cross_synthesis_input_est"] = cs_input_est
        token_stats["aggregation_input_est"] = agg_input_est
        token_stats["token_total"] = sum(
            v for k, v in token_stats.items() if not k.endswith("_input_est")
        )

        # Try robust JSON extraction first
        extracted = self._extract_json(aggre_output)
        if extracted:
            return extracted, report_count, token_stats

        # Fallback: try _parse_response + direct parse
        cleaned = self._parse_response(aggre_raw)
        try:
            json.loads(cleaned)
            return cleaned, report_count, token_stats
        except json.JSONDecodeError:
            pass

        # Last resort: wrap raw for debugging
        logging.warning(f"Final output is not valid JSON, wrapping in _raw")
        return json.dumps({"is_risk": "1", "risk_facts": [], "_raw": aggre_raw},
                          ensure_ascii=False), report_count, token_stats


# ============================================================
# Excel save functions
# ============================================================

def _write_sheet(excel_path: Path, sheet_name: str, df: pd.DataFrame):
    """Write or update a sheet, crash-safely.

    Old behaviour rewrote the workbook in place (``mode="a"``): if the process was
    killed mid-write (e.g. you Ctrl-C after the API runs out), the whole .xlsx
    could be truncated and every previously-saved case lost. This version reads
    the existing sheets, overlays the target one, writes everything to a temp file
    in the same directory, then atomically ``os.replace``s it onto the target — so
    a crash leaves either the old file or the new file intact, never a corrupt one.
    """
    xl_path = Path(excel_path)
    xl_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve any sibling sheets (Sheet1 / Eval_Metrics / Eval_Process).
    sheets: dict[str, pd.DataFrame] = {}
    if xl_path.exists():
        try:
            sheets = pd.read_excel(xl_path, sheet_name=None)
        except Exception as e:
            # The file exists but is unreadable right now (open in Excel, transient
            # lock, or genuine corruption). Refuse to overwrite — rebuilding from a
            # single sheet here would wipe the sibling sheets and every row we can't
            # see. Abort the write so the caller treats it as a failure and retries;
            # the existing file is left untouched.
            raise RuntimeError(
                f"_write_sheet: refusing to overwrite unreadable {xl_path.name} ({e}); "
                f"existing data left intact, case will be retried"
            ) from e
    sheets[sheet_name] = df

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".xlsx", dir=str(xl_path.parent), prefix=".cn_tmp_"
    )
    os.close(tmp_fd)
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for name, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=name, index=False)
        os.replace(tmp_path, xl_path)  # atomic on the same filesystem
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def save_agent_result(excel_path: Path, case_data: dict, xlsx_lock: asyncio.Lock) -> int:
    """Write/update Sheet1 (Agent results). Lock-protected for concurrent safety.

    Returns the Sheet1 row Id for cross-referencing in Eval sheets."""
    async with xlsx_lock:
        xl_path = Path(excel_path)
        if xl_path.exists():
            try:
                df = pd.read_excel(xl_path, sheet_name="Sheet1")
            except ValueError:
                df = pd.DataFrame(columns=["Id", "CaseId", "CompanyName", "Symbol", "InputPath",
                                           "ReportCount", "FraudInfo", "RiskAnalysis"])
        else:
            df = pd.DataFrame(columns=["Id", "CaseId", "CompanyName", "Symbol", "InputPath",
                                       "ReportCount", "FraudInfo", "RiskAnalysis"])

        case_id = str(case_data["CaseId"])
        if case_id in set(str(c) for c in df["CaseId"].values):
            idx = df[df["CaseId"].astype(str) == case_id].index[0]
            for col in ["CompanyName", "Symbol", "InputPath", "ReportCount", "FraudInfo", "RiskAnalysis"]:
                df.loc[idx, col] = case_data.get(col, "")
            sheet1_id = int(df.loc[idx, "Id"])
        else:
            sheet1_id = len(df) + 1
            new_row = {
                "Id": sheet1_id,
                "CaseId": case_id,
                "CompanyName": case_data.get("CompanyName", ""),
                "Symbol": case_data.get("Symbol", ""),
                "InputPath": case_data.get("InputPath", ""),
                "ReportCount": case_data.get("ReportCount", 0),
                "FraudInfo": case_data.get("FraudInfo", ""),
                "RiskAnalysis": case_data.get("RiskAnalysis", ""),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        _write_sheet(excel_path, "Sheet1", df)
        return sheet1_id


async def save_eval_result(excel_path: Path, case_id: str, eval_data: dict, xlsx_lock: asyncio.Lock,
                          sheet1_id: int = 0):
    """Write/update Eval_Metrics sheet (LLM judge results). Lock-protected for concurrent safety."""
    async with xlsx_lock:
        xl_path = Path(excel_path)
        columns = ["Id", "CaseId"] + [k for k in eval_data.keys() if k != "CaseId"]

        if xl_path.exists():
            try:
                df_ed = pd.read_excel(xl_path, sheet_name="Eval_Metrics")
            except ValueError:
                df_ed = pd.DataFrame(columns=columns)
        else:
            df_ed = pd.DataFrame(columns=columns)

        case_id_str = str(case_id)
        if "CaseId" in df_ed.columns and case_id_str in df_ed["CaseId"].values.astype(str):
            idx = df_ed[df_ed["CaseId"].astype(str) == case_id_str].index[0]
            if "Id" in df_ed.columns:
                df_ed.loc[idx, "Id"] = sheet1_id
            for k, v in eval_data.items():
                if k in df_ed.columns:
                    df_ed.loc[idx, k] = v
        else:
            new_row = {"Id": sheet1_id, "CaseId": case_id_str, **eval_data}
            df_ed = pd.concat([df_ed, pd.DataFrame([new_row])], ignore_index=True)

        _write_sheet(excel_path, "Eval_Metrics", df_ed)


async def save_eval_process(excel_path: Path, process_row: dict, xlsx_lock: asyncio.Lock,
                            sheet1_id: int = 0):
    """Write/update Eval_Process sheet (per-risk_fact LLM judge details)."""
    process_row = {**process_row, "Id": sheet1_id}
    async with xlsx_lock:
        xl_path = Path(excel_path)
        columns = ["Id", "CaseId", "CompanyName", "GT_Raw_Count", "GT_Dedup_Count",
                   "GT_Duplicates", "GT_Dedup_Issues",
                   "risk1", "risk2", "risk3", "risk4", "risk5",
                   "eval_1", "eval_2", "eval_3", "eval_4", "eval_5"]

        if xl_path.exists():
            try:
                df_ep = pd.read_excel(xl_path, sheet_name="Eval_Process")
            except ValueError:
                df_ep = pd.DataFrame(columns=columns)
        else:
            df_ep = pd.DataFrame(columns=columns)

        case_id_str = str(process_row["CaseId"])
        if "CaseId" in df_ep.columns and case_id_str in df_ep["CaseId"].values.astype(str):
            idx = df_ep[df_ep["CaseId"].astype(str) == case_id_str].index[0]
            for k, v in process_row.items():
                if k in df_ep.columns:
                    df_ep.loc[idx, k] = v
        else:
            df_ep = pd.concat([df_ep, pd.DataFrame([process_row])], ignore_index=True)

        _write_sheet(excel_path, "Eval_Process", df_ep)


async def write_summary_row(excel_path: Path, xlsx_lock: asyncio.Lock):
    """Compute summary from Eval_Metrics and append as last row."""
    async with xlsx_lock:
        xl_path = Path(excel_path)
        try:
            df = pd.read_excel(xl_path, sheet_name="Eval_Metrics")
        except ValueError:
            return

        if df.empty:
            return

        # Remove any existing summary row
        if "CaseId" in df.columns:
            df = df[df["CaseId"].astype(str) != "SUMMARY"]

        n = len(df)
        if n == 0:
            return

        # Macro averages
        macro_R_I = df["R_I"].mean()
        macro_P_I = df["P_I"].mean()
        macro_F1_I = df["F1_I"].mean()
        macro_R_E = df["R_E"].mean()

        # Micro averages
        total_gt = df["n_gt(dedup)_R_I_denom"].sum()
        total_rf = df["n_risk_facts(P_I_denom)"].sum()
        total_rt1_ri = df["n_risk_title_1(R_I_num)"].sum()
        total_rt1_pi = df["n_risk_title_1(P_I_num)"].sum()
        total_gt_ev = df["n_gt_evidence(R_E_denom)"].sum()
        total_ev1 = df["n_ev_chain_1(R_E_num)"].sum()
        total_dups = df["duplicates_removed"].sum()

        micro_R_I = total_rt1_ri / total_gt if total_gt > 0 else 0.0
        micro_P_I = total_rt1_pi / total_rf if total_rf > 0 else 0.0
        micro_F1 = 2 * micro_P_I * micro_R_I / (micro_P_I + micro_R_I) if (micro_P_I + micro_R_I) > 0 else 0.0
        micro_R_E = total_ev1 / total_gt_ev if total_gt_ev > 0 else 0.0

        summary = pd.DataFrame([{
            "CaseId": "SUMMARY",
            "CompanyName": f"N={n}",
            "n_gt(dedup)_R_I_denom": total_gt,
            "n_risk_facts(P_I_denom)": total_rf,
            "n_risk_title_1(R_I_num)": total_rt1_ri,
            "n_risk_title_1(P_I_num)": total_rt1_pi,
            "n_gt_evidence(R_E_denom)": total_gt_ev,
            "n_ev_chain_1(R_E_num)": total_ev1,
            "R_I": round(macro_R_I, 6),
            "P_I": round(macro_P_I, 6),
            "F1_I": round(macro_F1_I, 6),
            "R_E": round(macro_R_E, 6),
            "duplicates_removed": total_dups,
            "R_I_micro": round(micro_R_I, 6),
            "P_I_micro": round(micro_P_I, 6),
            "F1_I_micro": round(micro_F1, 6),
            "R_E_micro": round(micro_R_E, 6),
        }])
        df = pd.concat([df, summary], ignore_index=True)
        _write_sheet(excel_path, "Eval_Metrics", df)

        # Print summary
        print("\n" + "=" * 80)
        print("LLM JUDGE METRICS SUMMARY")
        print("=" * 80)
        print(f"  Cases: N={n}")
        print(f"  Macro: R_I={macro_R_I:.4f}  P_I={macro_P_I:.4f}  F1_I={macro_F1_I:.4f}  R_E={macro_R_E:.4f}")
        print(f"  Micro: R_I={micro_R_I:.4f}  P_I={micro_P_I:.4f}  F1_I={micro_F1:.4f}  R_E={micro_R_E:.4f}")
        print("=" * 80)


# ============================================================
# Case processing
# ============================================================

async def process_case(case: dict, model_name: str, prior: str, xlsx_lock: asyncio.Lock,
                       session: aiohttp.ClientSession, llm_sem: asyncio.Semaphore,
                       df_dataset: pd.DataFrame) -> dict | None:
    """Run Agent + LLM Judge eval on a single Chinese case."""
    case_id = case["case_id"]
    company = case["company"]
    symbol = case["symbol"]
    context_path = case["context_path"]
    raw_row = case["raw_row"]

    print(f"\n  CN-{case_id}: {company} ({symbol})")
    print(f"    Context: {context_path}")

    # Verify context file
    if not context_path or not context_path.exists():
        print(f"    SKIP: Context file not found")
        return None

    # Build GT for LLM judge from dataset row
    gt = build_gt_for_case(raw_row)
    gt_label_json = convert_gt_to_label_format(gt)
    dedup_stats = get_dedup_stats(str(raw_row.get("Term", "")))
    n_gt_dedup = len(gt["affected_accounts_dedup"])
    n_gt_evidence = sum(1 for loc in gt["fraud_locations"] if loc.get("evidence") and len(loc["evidence"]) >= 5)
    print(f"    GT: {n_gt_dedup} dedup issues, {n_gt_evidence} evidence entries")

    # Run Agent.
    # On ANY failure we return without saving, so the case stays out of Sheet1 and
    # the next run retries it — a flaky/exhausted API never gets frozen in as an
    # empty "success" that resume would skip forever.
    result_str = None
    report_count = 1
    token_stats: dict[str, int] = {}
    try:
        async with CnAgentRunner(model_name) as runner:
            result_str, report_count, token_stats = await runner.analyze(
                context_path, prior=prior, symbol=case.get("symbol", "")
            )
        parsed = json.loads(result_str)
        is_risk = parsed.get("is_risk", "?")
        n_facts = len(parsed.get("risk_facts", []))
        print(f"    Agent: is_risk={is_risk}, risk_facts={n_facts}")
        if token_stats:
            print(f"    Tokens: {token_stats}")
    except TerminalAPIError as e:
        print(f"    [STOP] Terminal API error (no balance / quota / auth) — "
              f"case NOT saved, run is stopping: {e}")
        return None
    except CallFailedError as e:
        print(f"    [RETRY] Agent call failed after retries — case NOT saved, "
              f"will be retried next run: {e}")
        return None
    except Exception as e:
        print(f"    [RETRY] Agent error — case NOT saved, will be retried next run: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not result_str:
        return None

    # Reached only on a genuine agent result → safe to persist (atomic write).
    sheet1_id = 0
    try:
        agent_data = {
            "CaseId": case_id,
            "CompanyName": company,
            "Symbol": symbol,
            "InputPath": str(context_path),
            "ReportCount": report_count,
            "FraudInfo": ";".join(gt.get("fraud_info", [])),
            "RiskAnalysis": result_str,
        }
        sheet1_id = await save_agent_result(AUDIT_RESULT_PATH, agent_data, xlsx_lock)
        print(f"    Agent result saved to Sheet1 (row Id={sheet1_id})")
    except Exception as e:
        print(f"    [RETRY] Save failed — case NOT marked done, will be retried next run: {e}")
        return None

    try:
        pred = json.loads(result_str) if isinstance(result_str, str) else result_str
    except json.JSONDecodeError:
        print(f"    SKIP: JSON parse failed for agent output")
        return None

    try:
        risk_facts = pred.get("risk_facts", [])
        n_risk_facts = min(len(risk_facts), 5)
        top_is_risk = str(pred.get("is_risk", ""))

        if n_risk_facts == 0:
            print(f"    No risk_facts to evaluate")
            metrics = compute_case_metrics({}, 0, n_gt_dedup, n_gt_evidence,
                                           dedup_stats["raw"], dedup_stats["dedup"])
            risk_cols = {f"risk{j+1}": "" for j in range(5)}
            eval_cols = {f"eval_{j+1}": "" for j in range(5)}
        else:
            # Evaluate each risk_fact with LLM judge (sequential within case)
            evals: dict[int, dict] = {}
            risk_cols: dict[str, str] = {}
            eval_cols: dict[str, str] = {}
            for j in range(5):
                if j < n_risk_facts:
                    rf = risk_facts[j]
                    risk_cols[f"risk{j+1}"] = json.dumps(rf, ensure_ascii=False)
                    print(f"    LLM Judge [{j+1}/{n_risk_facts}]: \"{rf.get('risk_title', '')[:60]}\"", end="", flush=True)
                    eval_result = await llm_eval_risk_fact(
                        session, llm_sem, rf, top_is_risk, gt_label_json,
                    )
                    evals[j] = eval_result
                    eval_cols[f"eval_{j+1}"] = json.dumps(eval_result, ensure_ascii=False)
                    score_str = "/".join(str(eval_result.get(k, 0)) for k in ["is_risk", "risk_title", "involved_report", "evidence_chain"])
                    print(f" -> [{score_str}]")
                else:
                    risk_cols[f"risk{j+1}"] = ""
                    eval_cols[f"eval_{j+1}"] = ""

            metrics = compute_case_metrics(evals, n_risk_facts, n_gt_dedup, n_gt_evidence,
                                           dedup_stats["raw"], dedup_stats["dedup"])

        # Save Eval_Process row (per-risk details)
        process_row = {
            "CaseId": case_id,
            "CompanyName": company,
            "GT_Raw_Count": dedup_stats["raw"],
            "GT_Dedup_Count": dedup_stats["dedup"],
            "GT_Duplicates": json.dumps(dedup_stats["duplicates"], ensure_ascii=False),
            "GT_Dedup_Issues": json.dumps(gt["affected_accounts_dedup"], ensure_ascii=False),
            **risk_cols,
            **eval_cols,
        }
        await save_eval_process(AUDIT_RESULT_PATH, process_row, xlsx_lock, sheet1_id)

        r_i = metrics["R_I"]
        p_i = metrics["P_I"]
        f1_i = metrics["F1_I"]
        r_e = metrics["R_E"]
        print(f"    Eval: P_I={p_i:.3f} R_I={r_i:.3f} F1={f1_i:.3f} R_E={r_e:.3f}")

        # Save eval result in Eval_Metrics format
        eval_data = {
            "CaseId": case_id,
            "CompanyName": company,
            **metrics,
        }
        # Merge token stats into eval_data
        if token_stats:
            for tk, tv in token_stats.items():
                eval_data[f"Tokens_{tk}"] = tv
        await save_eval_result(AUDIT_RESULT_PATH, case_id, eval_data, xlsx_lock, sheet1_id)
        print(f"    Eval result saved to Eval_Metrics")

        return {
            "CaseId": case_id,
            "CompanyName": company,
            "R_I": r_i,
            "F1_I": f1_i,
        }
    except Exception as e:
        print(f"    Eval ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Main
# ============================================================

def load_cn_cases(limit: int = 0) -> list[dict]:
    """Load Chinese cases from the dataset Excel."""
    df = pd.read_excel(DATASET_XLSX)
    if limit > 0:
        df = df.head(limit)

    cases: list[dict] = []
    for _, row in df.iterrows():
        context = str(row.get("Context", ""))
        if pd.isna(row.get("Context")) or not context.strip():
            continue

        context_filename = Path(context).name
        context_path = TXT_PREFIX / context_filename
        if not context_path.exists():
            continue

        label = _row_to_unified_label(row)
        report_paths = _split_multi_value(str(row.get("InputPdfPath", "")))
        cases.append({
            "case_id": str(row["Id"]),
            "company": str(row.get("CoFullName", "")),
            "symbol": str(row.get("Symbol", "")),
            "context_path": context_path,
            "report_count": len(report_paths),
            "fraud_report_name": str(row.get("FraudReportName", "")),
            "unified_label": label,
            "raw_row": row,
        })
    return cases


async def main():
    parser = argparse.ArgumentParser(description="Run Chinese Agent + metrics_new Eval on FinFraud dataset")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of cases to process (0 = no limit)")
    parser.add_argument("--reeval-only", action="store_true",
                        help="Only run eval on existing cases in Sheet1, skip Agent analysis")
    parser.add_argument("--model", type=str, default="deepseek-v4",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model name")
    parser.add_argument("--prior", type=str, default="cn", choices=["cn", "us", "cn_us"],
                        help="Prior subject source: cn (CN 15, default), us (US 15→CN), cn_us (CN+US merged→CN)")
    args = parser.parse_args()
    limit = args.limit
    reeval_only = args.reeval_only
    model_name = args.model
    prior = args.prior

    # Circuit breaker: tripped by the first terminal (no-balance/quota/auth) error.
    global STOP_EVENT
    STOP_EVENT = asyncio.Event()

    # Build output path: append model name + prior suffix
    global AUDIT_RESULT_PATH
    stem = AUDIT_RESULT_PATH.stem
    suffix_parts = [model_name]
    if prior != "cn":
        suffix_parts.append(prior)
    AUDIT_RESULT_PATH = AUDIT_RESULT_PATH.parent / f"{stem}_{'_'.join(suffix_parts)}.xlsx"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if reeval_only:
        print("=" * 80)
        print("Re-evaluation Only - Run metrics_new evaluator on existing cases in Sheet1")
        print("=" * 80)

        if not AUDIT_RESULT_PATH.exists():
            print(f"ERROR: {AUDIT_RESULT_PATH} does not exist")
            return

        df = pd.read_excel(AUDIT_RESULT_PATH, sheet_name="Sheet1")
        existing_cases = df.to_dict("records")
        print(f"\nFound {len(existing_cases)} cases in Sheet1")

        existing_eval_ids = set()
        try:
            df_ed = pd.read_excel(AUDIT_RESULT_PATH, sheet_name="Eval_Metrics")
            existing_eval_ids = set(str(cid) for cid in df_ed["CaseId"].tolist())
            print(f"Found {len(existing_eval_ids)} cases in Eval_Metrics (will skip)")
        except ValueError:
            print("Eval_Metrics sheet does not exist yet")

        to_reeval = [c for c in existing_cases if str(c["CaseId"]) not in existing_eval_ids]
        print(f"Cases to re-evaluate: {len(to_reeval)}")

        if limit > 0:
            to_reeval = to_reeval[:limit]
            print(f"Limit set: processing first {len(to_reeval)} cases")

        if not to_reeval:
            print("No cases to re-evaluate!")
            return

        case_sem = asyncio.Semaphore(3)
        llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)
        xlsx_lock = asyncio.Lock()

        # Load full dataset for GT lookup
        df_full = pd.read_excel(DATASET_XLSX)

        async with aiohttp.ClientSession() as session:
            async def reeval_one(c, idx):
                async with case_sem:
                    case_id = c["CaseId"]
                    risk_analysis = c.get("RiskAnalysis", "")
                    if not risk_analysis:
                        print(f"\n  CN-{case_id}: SKIP - No RiskAnalysis")
                        return None

                    try:
                        pred = json.loads(risk_analysis) if isinstance(risk_analysis, str) else risk_analysis
                    except json.JSONDecodeError:
                        print(f"\n  CN-{case_id}: SKIP - Invalid JSON")
                        return None

                    # Find GT in dataset
                    row_match = df_full[df_full["Id"].astype(str) == str(case_id)]
                    if row_match.empty:
                        print(f"\n  CN-{case_id}: SKIP - Not found in dataset")
                        return None

                    row = row_match.iloc[0]
                    print(f"\n  CN-{case_id}: {c['CompanyName']}")

                    try:
                        gt = build_gt_for_case(row)
                        gt_label_json = convert_gt_to_label_format(gt)
                        dedup_stats = get_dedup_stats(str(row.get("Term", "")))
                        n_gt_dedup = len(gt["affected_accounts_dedup"])
                        n_gt_evidence = sum(1 for loc in gt["fraud_locations"]
                                           if loc.get("evidence") and len(loc["evidence"]) >= 5)

                        risk_facts = pred.get("risk_facts", [])
                        n_risk_facts = min(len(risk_facts), 5)
                        top_is_risk = str(pred.get("is_risk", ""))

                        if n_risk_facts == 0:
                            metrics = compute_case_metrics({}, 0, n_gt_dedup, n_gt_evidence,
                                                           dedup_stats["raw"], dedup_stats["dedup"])
                            risk_cols = {f"risk{j+1}": "" for j in range(5)}
                            eval_cols = {f"eval_{j+1}": "" for j in range(5)}
                        else:
                            evals: dict[int, dict] = {}
                            risk_cols: dict[str, str] = {}
                            eval_cols: dict[str, str] = {}
                            for j in range(5):
                                if j < n_risk_facts:
                                    rf = risk_facts[j]
                                    risk_cols[f"risk{j+1}"] = json.dumps(rf, ensure_ascii=False)
                                    print(f"    LLM Judge [{j+1}/{n_risk_facts}]: \"{rf.get('risk_title', '')[:60]}\"", end="", flush=True)
                                    eval_result = await llm_eval_risk_fact(
                                        session, llm_sem, rf, top_is_risk, gt_label_json,
                                    )
                                    evals[j] = eval_result
                                    eval_cols[f"eval_{j+1}"] = json.dumps(eval_result, ensure_ascii=False)
                                    score_str = "/".join(str(eval_result.get(k, 0)) for k in ["is_risk", "risk_title", "involved_report", "evidence_chain"])
                                    print(f" -> [{score_str}]")
                                else:
                                    risk_cols[f"risk{j+1}"] = ""
                                    eval_cols[f"eval_{j+1}"] = ""
                            metrics = compute_case_metrics(evals, n_risk_facts, n_gt_dedup, n_gt_evidence,
                                                           dedup_stats["raw"], dedup_stats["dedup"])

                        # Save Eval_Process row
                        sheet1_id = int(c.get("Id", 0))
                        process_row = {
                            "CaseId": case_id,
                            "CompanyName": c.get("CompanyName", ""),
                            "GT_Raw_Count": dedup_stats["raw"],
                            "GT_Dedup_Count": dedup_stats["dedup"],
                            "GT_Duplicates": json.dumps(dedup_stats["duplicates"], ensure_ascii=False),
                            "GT_Dedup_Issues": json.dumps(gt["affected_accounts_dedup"], ensure_ascii=False),
                            **risk_cols,
                            **eval_cols,
                        }
                        await save_eval_process(AUDIT_RESULT_PATH, process_row, xlsx_lock, sheet1_id)

                        p_i = metrics["P_I"]
                        r_i = metrics["R_I"]
                        f1_i = metrics["F1_I"]
                        r_e = metrics["R_E"]
                        print(f"    Eval: P_I={p_i:.3f} R_I={r_i:.3f} F1={f1_i:.3f} R_E={r_e:.3f}")

                        eval_data = {
                            "CompanyName": c.get("CompanyName", ""),
                            **metrics,
                        }
                        await save_eval_result(AUDIT_RESULT_PATH, case_id, eval_data, xlsx_lock, sheet1_id)
                        print(f"    Eval result saved to Eval_Metrics")
                        return True
                    except Exception as e:
                        print(f"    Eval ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        return None

            tasks = [asyncio.create_task(reeval_one(c, i)) for i, c in enumerate(to_reeval)]
            results = await asyncio.gather(*tasks)
            processed = sum(1 for r in results if r)

        print(f"\n{'='*80}")
        print(f"DONE: Re-evaluated {processed} cases")
        print("=" * 80)

        # Write summary row
        await write_summary_row(AUDIT_RESULT_PATH, xlsx_lock)
        return

    # Normal mode: run Agent + Eval
    print("=" * 80)
    print(f"Chinese Agent (Hierarchical RAG) + metrics_new Evaluator")
    print(f"Prior source: {prior}")
    print("=" * 80)

    # Load cases
    all_cases = load_cn_cases()
    print(f"\nTotal valid cases in dataset: {len(all_cases)}")

    # Skip already processed
    existing_ids = set()
    if AUDIT_RESULT_PATH.exists():
        try:
            existing_df = pd.read_excel(AUDIT_RESULT_PATH, sheet_name="Sheet1")
            existing_ids = set(str(cid) for cid in existing_df["CaseId"].tolist())
            print(f"Already processed: {len(existing_ids)} cases")
        except ValueError:
            print("Sheet1 does not exist yet")

    unprocessed = [c for c in all_cases if str(c["case_id"]) not in existing_ids]
    print(f"New cases to process: {len(unprocessed)}")

    if limit > 0:
        unprocessed = unprocessed[:limit]
        print(f"Limit set: processing first {len(unprocessed)} cases")

    if not unprocessed:
        print("Nothing to do!")
        return

    print(f"\nModel: {model_name}")
    for c in unprocessed[:5]:
        print(f"  - CN-{c['case_id']}: {c['company']}")
    if len(unprocessed) > 5:
        print(f"  ... and {len(unprocessed) - 5} more")

    print(f"\n{'='*80}")
    print(f"Starting processing of {len(unprocessed)} cases")
    print("=" * 80)

    case_sem = asyncio.Semaphore(3)
    llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)
    xlsx_lock = asyncio.Lock()
    df_dataset = pd.read_excel(DATASET_XLSX)

    async with aiohttp.ClientSession() as session:
        async def run_one(case, idx):
            async with case_sem:
                # If a terminal API error already tripped the breaker, stop pulling
                # new cases instead of hammering a dead/empty quota.
                if STOP_EVENT is not None and STOP_EVENT.is_set():
                    return None
                print(f"\n[{idx+1}/{len(unprocessed)}] Processing CN-{case['case_id']}")
                result = await process_case(case, model_name, prior, xlsx_lock, session, llm_sem, df_dataset)
                if result:
                    print(f"  Done: R_I={result.get('R_I', 0):.3f}, F1={result.get('F1_I', 0):.3f}")
                return result

        tasks = [asyncio.create_task(run_one(c, i)) for i, c in enumerate(unprocessed)]
        results = await asyncio.gather(*tasks)
        processed_count = sum(1 for r in results if r)

    stopped = STOP_EVENT is not None and STOP_EVENT.is_set()

    # Final summary
    print(f"\n{'='*80}")
    if stopped:
        print("STOPPED EARLY: terminal API error (no balance / quota / auth).")
        print("Already-finished cases are saved; unfinished ones were NOT saved —")
        print("top up the account and re-run the same command to resume from where it stopped.")
    print(f"DONE: Processed {processed_count} new cases")
    if AUDIT_RESULT_PATH.exists():
        df_final = pd.read_excel(AUDIT_RESULT_PATH, sheet_name="Sheet1")
        print(f"Total in {AUDIT_RESULT_PATH.name} Sheet1: {len(df_final)}")
    print("=" * 80)

    # Write summary row to Eval_Metrics
    await write_summary_row(AUDIT_RESULT_PATH, xlsx_lock)


if __name__ == "__main__":
    asyncio.run(main())
