"""
4-dimension LLM Judge evaluation for US cases.

Evaluates each risk_fact against ground truth on 4 binary dimensions:
  is_risk, risk_title, involved_report, evidence_chain

Adapted from LLM-FinRisk/LLM-FinRisk/evaluation/llm_judge.py
with US-specific GT format handling.
"""

import asyncio
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import aiohttp
import tiktoken

# Import legacy evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent
                       / "LLM-FinRisk" / "LLM-FinRisk" / "evaluation"))
from metrics import fraudcase_to_unified

def _extract_first_json(text: str) -> str:
    """Extract the first balanced JSON object from text by tracking brace depth."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


ENCODING_NAME = "cl100k_base"
LLM_CONCURRENCY = 3
LLM_MAX_RETRIES = 5
LLM_REQUEST_DELAY = 1

_SEPARATOR = "\n next \n"

# TODO: Set your API key via environment variable before running.
# Original key backed up in api_keys_backup.txt (EXCLUDED from git).
JUDGE_API_CONFIG = {
    "url": "https://api.minimax.chat/v1/chat/completions",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY",
    },
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for financial fraud detection. Your task is to compare a predicted risk fact against the ground truth (GT) for a financial fraud case and determine whether the prediction matches the GT.

You must evaluate FOUR binary dimensions:

1. **is_risk** (0 or 1): The case-level judgment — if prediction says is_risk="0" (no risk), but GT says this IS a fraud case (is_fraud=1), then is_risk=0 (wrong). If prediction says is_risk="1" and GT is indeed fraud, then is_risk=1. If both agree on no fraud, is_risk=1 (correct negative). In short: is_risk=1 if the prediction correctly identified whether the case is fraudulent or not.

2. **risk_title** (0 or 1): Does the predicted risk_title match ANY of the GT accounting issues? Match criteria: the accounting subject (e.g., "Revenue recognition", "Accounts Receivable overstatement") overlaps between prediction and any GT issue. Semantic equivalence counts as a match.

3. **involved_report** (0 or 1): Does the predicted involved_report match ANY of the GT fraud report names? Year-only matching is acceptable (e.g., "2000" matches "2000年年度报告").

4. **evidence_chain** (0 or 1): Does the predicted evidence_chain text have ANY factual overlap with the GT evidence? At least one evidence point or specific figure/event must match between prediction and GT.

Output ONLY a JSON object with these four keys:
{"is_risk": 0 or 1, "risk_title": 0 or 1, "involved_report": 0 or 1, "evidence_chain": 0 or 1}
"""


def _truncate_tokens(text: str, max_tokens: int) -> str:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])


# ---- GT builder for US cases ----

def build_gt_for_us_case(gt_case: dict) -> dict:
    """Convert a US case JSON dict into the GT format expected by the LLM judge.

    Args:
        gt_case: Raw US case dict (case_LR-XXXXX.json).

    Returns:
        Dict compatible with build_gt_for_case() output.
    """
    unified = fraudcase_to_unified(gt_case, use_accounting_issues=True)

    # Build affected_accounts_dedup (from unified affected_accounts)
    affected_accounts = unified.get("affected_accounts", [])
    seen_acc: set[str] = set()
    dedup_accounts: list[str] = []
    for acc in affected_accounts:
        name = acc.get("account", "").strip()
        if name and name not in seen_acc:
            seen_acc.add(name)
            dedup_accounts.append(name)

    # Build fraud_locations list
    fraud_locations = unified.get("fraud_locations", [])

    # Count GT evidence
    n_gt_evidence = sum(
        1 for loc in fraud_locations
        if loc.get("evidence") and len(str(loc["evidence"])) >= 5
    )

    is_fraud = int(unified.get("is_fraud", 1))

    return {
        "case_id": unified.get("case_id", ""),
        "is_fraud": is_fraud,
        "affected_accounts_dedup": dedup_accounts,
        "fraud_locations": fraud_locations,
        "_n_gt_evidence": n_gt_evidence,
    }


def convert_gt_to_label_format(gt: dict) -> str:
    """Serialize GT dict to JSON string format for the LLM judge prompt.

    Args:
        gt: Output of build_gt_for_us_case().

    Returns:
        JSON string with fraud_info, affected_accounts, fraud_locations.
    """
    fraud_info = gt.get("fraud_info", [])
    affected_accounts = [
        {"account": a, "gaap_codification": ""}
        for a in gt.get("affected_accounts_dedup", [])
    ]
    fraud_locations = []
    for loc in gt.get("fraud_locations", []):
        fraud_locations.append({
            "report": loc.get("report", ""),
            "summary": loc.get("summary", ""),
            "evidence": loc.get("evidence", ""),
        })

    label = {
        "is_fraud": gt.get("is_fraud", 1),
        "fraud_info": fraud_info,
        "affected_accounts": affected_accounts,
        "fraud_locations": fraud_locations,
    }
    return json.dumps(label, ensure_ascii=False)


# ---- LLM Judge call ----

async def llm_eval_risk_fact(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    risk_fact: dict,
    top_is_risk: str,
    gt_label_json: str,
) -> dict:
    """Evaluate one risk_fact against ground truth via LLM judge.

    Returns {is_risk, risk_title, involved_report, evidence_chain} dict.
    """
    rf_json = json.dumps(risk_fact, ensure_ascii=False)
    user_prompt = f"""Prediction (top-level is_risk={top_is_risk}):
{rf_json}

Ground Truth:
{gt_label_json}

Evaluate the FOUR dimensions and output ONLY the JSON result."""
    user_prompt = _truncate_tokens(user_prompt, 8000)

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            async with sem:
                async with session.post(
                    JUDGE_API_CONFIG["url"],
                    headers=JUDGE_API_CONFIG["headers"],
                    json={
                        "model": "MiniMax-M2.5",
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                    },
                ) as resp:
                    if resp.status == 429:
                        wait = 10 * (attempt + 1)
                        logging.warning(f"Judge 429 — waiting {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)
                        continue

                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    cleaned = content.replace("```json", "").replace("```", "").strip()
                    cleaned = _extract_first_json(cleaned)
                    result = json.loads(cleaned)
                    # Validate all 4 keys exist
                    for k in ["is_risk", "risk_title", "involved_report", "evidence_chain"]:
                        if k not in result:
                            result[k] = 0
                    return result

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Judge parse error (attempt {attempt+1}): {e}")
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            logging.error(f"Judge call error (attempt {attempt+1}): {e}")
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(min(30, 5 * (attempt + 1)))

    return {"is_risk": 0, "risk_title": 0, "involved_report": 0, "evidence_chain": 0}


# ---- Metrics computation ----

def compute_case_metrics(
    evals: dict[int, dict],
    n_risk_facts: int,
    n_gt_dedup: int,
    n_gt_evidence: int,
    raw_count: int = 0,
    dedup_count: int = 0,
) -> dict:
    """Compute R_I, P_I, F1_I, R_E for one case.

    Args:
        evals: {index: {is_risk, risk_title, involved_report, evidence_chain}} from LLM judge.
        n_risk_facts: Number of predicted risk_facts (max 5).
        n_gt_dedup: Number of deduplicated GT accounting subjects.
        n_gt_evidence: Number of GT evidence chains (length >= 5).
        raw_count: Raw GT count before dedup.
        dedup_count: Deduplicated GT count.

    Returns:
        Dict with all metrics.
    """
    n_rt1_ri = 0
    n_rt1_pi = 0
    n_ec1 = 0

    if n_risk_facts == 0:
        return {
            "R_I": 0.0, "P_I": 0.0, "F1_I": 0.0, "R_E": 0.0,
            "n_gt(dedup)_R_I_denom": n_gt_dedup,
            "n_risk_facts(P_I_denom)": 0,
            "n_risk_title_1(R_I_num)": 0,
            "n_risk_title_1(P_I_num)": 0,
            "n_gt_evidence(R_E_denom)": n_gt_evidence,
            "n_ev_chain_1(R_E_num)": 0,
            "duplicates_removed": max(0, raw_count - dedup_count),
        }

    # Collect unique hit indices for R_I dedup
    ri_seen: set[int] = set()
    for j, ev in evals.items():
        if ev.get("risk_title", 0) == 1:
            n_rt1_pi += 1
        if ev.get("evidence_chain", 0) == 1:
            n_ec1 += 1

    n_rt1_ri = n_rt1_pi  # risk_title=1 count same for both num/dedup at case level

    r_i = min(n_rt1_ri / n_gt_dedup, 1.0) if n_gt_dedup > 0 else 0.0
    p_i = (n_rt1_pi / n_risk_facts) if n_risk_facts > 0 else 0.0
    f1 = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
    r_e = min(n_ec1 / n_gt_evidence, 1.0) if n_gt_evidence > 0 else 0.0

    return {
        "R_I": round(r_i, 6), "P_I": round(p_i, 6), "F1_I": round(f1, 6),
        "R_E": round(r_e, 6),
        "n_gt(dedup)_R_I_denom": n_gt_dedup,
        "n_risk_facts(P_I_denom)": n_risk_facts,
        "n_risk_title_1(R_I_num)": n_rt1_ri,
        "n_risk_title_1(P_I_num)": n_rt1_pi,
        "n_gt_evidence(R_E_denom)": n_gt_evidence,
        "n_ev_chain_1(R_E_num)": n_ec1,
        "duplicates_removed": max(0, raw_count - dedup_count),
    }
