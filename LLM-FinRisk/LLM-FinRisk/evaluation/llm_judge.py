"""
LLM Judge module for per-risk_fact 4-dimension binary evaluation.

Evaluates each risk_fact against GT label using minimax-m2.5 LLM judge,
then computes paper-formula metrics (R_I, P_I, F1_I, R_E) from the binary results.

Extracted from scripts/add_paper_eval_sheets.py for reuse by Agent and Baseline scripts.
"""

import asyncio
import json
import re
from collections import OrderedDict

import aiohttp
import pandas as pd

# --- Constants ---
SEPARATOR = "\n next \n"

# --- LLM API Config ---
MINIMAX_API_URL = "https://api.minimax.chat/v1/chat/completions"
# TODO: Set your API key via environment variable.
# Original key backed up in api_keys_backup.txt (EXCLUDED from git).
MINIMAX_API_KEY = "YOUR_API_KEY"
MINIMAX_MODEL = "MiniMax-M2.5"
LLM_CONCURRENCY = 3
LLM_MAX_RETRIES = 5
LLM_REQUEST_DELAY = 1  # seconds between requests to avoid rate limiting

# ============================================================
# LLM Judge System Prompt
# ============================================================

LLM_JUDGE_SYSTEM_PROMPT = """### **角色**
你是财务造假检测模型的专家评估者。

---

### **任务**
给你的是推理模型的输出和标签，格式为JSON。
内容分别在`<output>`和`<label>`中。
请评估模型输出与标签的匹配程度，并输出评估结果。

---

### **输入格式**
#### **`<label_format>`**
```json
{
  "IsFraud": "",
  "FraudInfo": [],
  "FraudLoc": [
    {
      "Report": "",
      "Summary": "",
      "Evidence": ""
    }
  ]
}
```

#### **字段说明**
1. **`IsFraud`**
   - 值：`0`或`1`。
   - 含义：
     - `0`：完全不涉及财务报告造假问题。
     - `1`：存在明确或直接的财务报告造假证据问题。

2. **`FraudInfo`**
   - 值：字符串列表。
   - 含义：
     - 若`IsFraud`为`0`：列表包含一个字符串，说明不涉及造假的原因。
     - 若`IsFraud`为`1`：列表包含一个或多个字符串，描述具体的财务报告造假问题。

3. **`FraudLoc`**
   - 值：字典列表。
   - 含义：
     - 若`IsFraud`为`0`：列表包含一个字典，所有字段值为空字符串（`{"Report": "", "Summary": "", "Evidence": ""}`）。
     - 若`IsFraud`为`1`：列表包含一个或多个字典，描述具体的造假报告、摘要和证据。

---

#### **`<output_format>`**
```json
{
  "is_risk": "",
  "risk_facts": [
    {
      "risk_title": "",
      "involved_report": "",
      "evidence_chain": [
        {
          "point": "",
          "analysis": ""
        }
      ]
    }
  ]
}
```

#### **字段说明**
1. **`is_risk`**
   - 值：`0`或`1`。
   - 含义：
     - `1`：财报存在财务造假风险。
     - `0`：财报无财务造假风险。

2. **`risk_title`**
   - 值：字符串。
   - 含义：财务造假问题的核心内容，例如"虚构应收账款收回"。

3. **`involved_report`**
   - 值：字符串。
   - 含义：造假问题涉及的具体报告名称，例如"2021年年度报告"。

4. **`evidence_chain`**
   - 值：字典列表。
   - 含义：对应造假问题的直接或间接的证据推理链描述，包括：
     - `point`：可疑点概要。
     - `analysis`：详细推理过程。

---

### **评估指标**
1. **`is_risk`**
   - 评估模型的`is_risk`是否与标签的`IsFraud`匹配。
   - 规则：
     - 若`IsFraud`为`1`，则`is_risk`应为`1`。
     - 若`IsFraud`为`0`，则`is_risk`应为`0`。
   - 输出：`0`（不匹配）或`1`（匹配）。

2. **`risk_title`**
   - 若`is_risk`为`1`，评估模型的`risk_title`与标签的`FraudLoc`中的`Summary`的相似度。
   - 规则：
     - 检查模型输出的`risk_title`是否与`FraudLoc`中的任意一个`Summary`字符串高度相关。
   - 输出：`0`（不匹配）或`1`（匹配）。

3. **`involved_report`**
   - 若`is_risk`为`1`，评估模型的`involved_report`是否与标签的`FraudLoc`中的`Report`匹配。
   - 规则：
     - 检查模型输出的`involved_report`是否与`FraudLoc`中的任意一个`Report`字段一致。
   - 输出：`0`（不匹配）或`1`（匹配）。

4. **`evidence_chain`**
   - 若`is_risk`为`1`，评估模型的`evidence_chain`与标签的`FraudLoc`中的`Evidence`的相似度。
   - 规则：
     - 检查模型输出的`evidence_chain`是否与`FraudLoc`中的`Evidence`字段内容一致或高度相关。
   - 输出：`0`（不匹配）或`1`（匹配）。

---

### **输出格式**
```json
{
  "is_risk": 0或1,
  "risk_title": 0或1,
  "involved_report": 0或1,
  "evidence_chain": 0或1
}
```
### **示例**
#### **输入**
```json
<label>
{
  "IsFraud": "1",
  "FraudInfo": ["虚构应收账款收回"],
  "FraudLoc": [
    {
      "Report": "2021年年度报告",
      "Summary": "通过虚构应收账款收回虚增利润",
      "Evidence": "应收账款明细表显示异常波动"
    }
  ]
}
</label>

<output>
{
  "is_risk": "1",
  "risk_facts": [
    {
      "risk_title": "虚构应收账款收回",
      "involved_report": "2021年年度报告",
      "evidence_chain": [
        {
          "point": "应收账款明细表显示异常波动",
          "analysis": "应收账款明细表在2021年第四季度出现异常波动，与历史数据不符"
        }
      ]
    }
  ]
}
</output>
```

#### **输出**
```json
{
  "is_risk": 1,
  "risk_title": 1,
  "involved_report": 1,
  "evidence_chain": 1
}
```
"""


# ============================================================
# GT processing helpers
# ============================================================

def split_multi_value(value) -> list[str]:
    """Split \\n next \\n separated multi-value into list, deduplicating."""
    if pd.isna(value) or not str(value).strip():
        return []
    parts = [s.strip() for s in str(value).split(SEPARATOR) if s.strip()]
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def get_dedup_stats(terms_str: str) -> dict:
    """Get dedup stats for a case's Term column."""
    if pd.isna(terms_str) or not str(terms_str).strip():
        return {"raw": 0, "dedup": 0, "duplicates": [], "dedup_list": []}

    raw_parts = [s.strip() for s in str(terms_str).split(SEPARATOR) if s.strip()]
    seen: OrderedDict[str, bool] = OrderedDict()
    duplicates: list[str] = []
    for p in raw_parts:
        if p in seen:
            duplicates.append(p)
        else:
            seen[p] = True
    return {
        "raw": len(raw_parts),
        "dedup": len(seen),
        "duplicates": duplicates,
        "dedup_list": list(seen.keys()),
    }


def build_gt_for_case(row: pd.Series) -> dict:
    """Build deduplicated GT label for a single case from the dataset row."""
    terms_dedup = split_multi_value(row.get("Term", ""))
    reports_dedup = split_multi_value(row.get("Report", ""))
    summaries_dedup = split_multi_value(row.get("Summary", ""))
    evidences_dedup = split_multi_value(row.get("Evidence", ""))

    n = max(len(reports_dedup), len(terms_dedup), len(summaries_dedup), len(evidences_dedup))
    fraud_locations: list[dict] = []
    for i in range(n):
        rpt = reports_dedup[i] if i < len(reports_dedup) else ""
        trm = terms_dedup[i] if i < len(terms_dedup) else ""
        smy = summaries_dedup[i] if i < len(summaries_dedup) else ""
        evd = evidences_dedup[i] if i < len(evidences_dedup) else ""
        fraud_locations.append({
            "report": rpt,
            "term": trm,
            "summary": smy,
            "evidence": evd,
        })

    return {
        "case_id": str(row.get("Id", "")),
        "is_fraud": int(row.get("IsFraud", 0)),
        "fraud_info": split_multi_value(row.get("FraudInfo", "")),
        "affected_accounts_dedup": terms_dedup,
        "fraud_locations": fraud_locations,
    }


# ============================================================
# LLM Judge helpers
# ============================================================

def convert_gt_to_label_format(gt: dict) -> str:
    """Convert GT dict to old evaluation.py label JSON format."""
    label = {
        "IsFraud": str(gt["is_fraud"]),
        "FraudInfo": gt.get("fraud_info", []),
        "FraudLoc": [
            {
                "Report": loc.get("report", ""),
                "Summary": loc.get("summary", ""),
                "Evidence": loc.get("evidence", ""),
            }
            for loc in gt.get("fraud_locations", [])
        ],
    }
    return json.dumps(label, ensure_ascii=False)


def build_risk_output_json(risk_fact: dict, top_is_risk: str) -> str:
    """Build single risk_fact output JSON in old evaluation.py format."""
    output = {
        "is_risk": str(top_is_risk),
        "risk_facts": [
            {
                "risk_title": risk_fact.get("risk_title", ""),
                "involved_report": risk_fact.get("involved_report", ""),
                "evidence_chain": risk_fact.get("evidence_chain", []),
            }
        ],
    }
    return json.dumps(output, ensure_ascii=False)


def parse_eval_result(content: str) -> dict | None:
    """Parse LLM judge response into {is_risk, risk_title, involved_report, evidence_chain} dict."""
    if not content:
        return None
    # Strip thinking tags
    think_end = content.find("</think>")
    if think_end != -1:
        content = content[think_end + 8:].strip()
    # Strip code blocks
    content = content.replace("```json", "").replace("```", "").strip()
    # Try direct parse
    try:
        result = json.loads(content)
        return {
            "is_risk": int(result.get("is_risk", 0)),
            "risk_title": int(result.get("risk_title", 0)),
            "involved_report": int(result.get("involved_report", 0)),
            "evidence_chain": int(result.get("evidence_chain", 0)),
        }
    except json.JSONDecodeError:
        pass
    # Try regex extraction
    try:
        match = re.search(r'({[\s\S]*})', content)
        if match:
            result = json.loads(match.group(1))
            return {
                "is_risk": int(result.get("is_risk", 0)),
                "risk_title": int(result.get("risk_title", 0)),
                "involved_report": int(result.get("involved_report", 0)),
                "evidence_chain": int(result.get("evidence_chain", 0)),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return None


async def llm_eval_risk_fact(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    risk_fact: dict,
    top_is_risk: str,
    gt_label_json: str,
) -> dict:
    """Evaluate a single risk_fact against GT using LLM judge. Returns eval dict."""
    output_json = build_risk_output_json(risk_fact, top_is_risk)
    user_prompt = f"""请根据以下输入内容进行评估：

<label>
{gt_label_json}
</label>

<output>
{output_json}
</output>"""

    payload = {
        "model": MINIMAX_MODEL,
        "messages": [
            {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
    }

    for attempt in range(LLM_MAX_RETRIES):
        try:
            async with sem:
                async with session.post(
                    MINIMAX_API_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        result = parse_eval_result(content)
                        if result is not None:
                            await asyncio.sleep(LLM_REQUEST_DELAY)
                            return result
                    elif resp.status == 429:
                        text = await resp.text()
                        wait = 10 * (attempt + 1)
                        print(f"    LLM 429 rate limited (attempt {attempt+1}), waiting {wait}s: {text[:100]}")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        text = await resp.text()
                        print(f"    LLM HTTP {resp.status} (attempt {attempt+1}): {text[:150]}")
        except Exception as e:
            print(f"    LLM error (attempt {attempt+1}): {e}")

        if attempt < LLM_MAX_RETRIES - 1:
            await asyncio.sleep(5 * (attempt + 1))

    # All retries failed, return default (all zeros)
    print(f"    WARNING: All {LLM_MAX_RETRIES} LLM retries failed, returning all zeros")
    return {"is_risk": 0, "risk_title": 0, "involved_report": 0, "evidence_chain": 0}


# ============================================================
# Metric computation
# ============================================================

def compute_case_metrics(
    evals: dict[int, dict],
    n_risk_facts: int,
    n_gt_dedup: int,
    n_gt_evidence: int,
    dedup_raw: int = 0,
    dedup_dedup: int = 0,
) -> dict:
    """Compute paper-formula metrics for one case from LLM judge results.

    Args:
        evals: dict mapping rf_idx -> {"is_risk": int, "risk_title": int, ...}
        n_risk_facts: number of risk_facts (P_I denominator)
        n_gt_dedup: number of deduplicated GT issues (R_I denominator)
        n_gt_evidence: number of GT entries with non-empty evidence (R_E denominator)
        dedup_raw: raw GT Term count before dedup
        dedup_dedup: GT Term count after dedup

    Returns:
        dict with R_I, P_I, F1_I, R_E and numerator/denominator fields
    """
    n_risk_title_1 = 0
    n_ev_chain_1 = 0

    for j in range(n_risk_facts):
        eval_result = evals.get(j, {"is_risk": 0, "risk_title": 0, "involved_report": 0, "evidence_chain": 0})
        if eval_result.get("risk_title", 0) == 1:
            n_risk_title_1 += 1
        if eval_result.get("evidence_chain", 0) == 1:
            n_ev_chain_1 += 1

    # R_I: cap risk_title=1 at n_gt_dedup
    r_i_num = min(n_risk_title_1, n_gt_dedup)
    r_i = r_i_num / n_gt_dedup if n_gt_dedup > 0 else 0.0

    # P_I: risk_title=1 / n_risk_facts
    p_i = n_risk_title_1 / n_risk_facts if n_risk_facts > 0 else 0.0

    # F1_I
    f1_i = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0

    # R_E: cap evidence_chain=1 at n_gt_evidence
    r_e_num = min(n_ev_chain_1, n_gt_evidence)
    r_e = r_e_num / n_gt_evidence if n_gt_evidence > 0 else 0.0

    return {
        "n_gt(dedup)_R_I_denom": n_gt_dedup,
        "n_risk_facts(P_I_denom)": n_risk_facts,
        "n_risk_title_1(R_I_num)": r_i_num,
        "n_risk_title_1(P_I_num)": n_risk_title_1,
        "n_gt_evidence(R_E_denom)": n_gt_evidence,
        "n_ev_chain_1(R_E_num)": r_e_num,
        "R_I": round(r_i, 6),
        "P_I": round(p_i, 6),
        "F1_I": round(f1_i, 6),
        "R_E": round(r_e, 6),
        "duplicates_removed": max(0, dedup_raw - dedup_dedup),
        "R_I_micro": "",
        "P_I_micro": "",
        "F1_I_micro": "",
        "R_E_micro": "",
    }


# ============================================================
# US case GT adapter
# ============================================================

def build_gt_for_us_case(gt_case: dict) -> dict:
    """Convert US JSON case to GT format compatible with convert_gt_to_label_format()
    and compute_case_metrics().

    Uses fraudcase_to_unified() to convert US JSON → unified format,
    then extracts the fields needed by the LLM judge pipeline.
    """
    from metrics import fraudcase_to_unified

    affected_accounts = gt_case.get("affected_accounts", [])
    use_accounting = len(affected_accounts) > 0

    unified = fraudcase_to_unified(gt_case, use_accounting_issues=use_accounting)

    fraud_info = unified.get("fraud_info", [])
    fraud_locations = unified.get("fraud_locations", [])
    n_gt_evidence = sum(
        1 for loc in fraud_locations
        if loc.get("evidence") and len(loc["evidence"]) >= 5
    )

    return {
        "case_id": unified.get("case_id", ""),
        "is_fraud": unified.get("is_fraud", 0),
        "fraud_info": fraud_info,
        "affected_accounts_dedup": list(dict.fromkeys(fraud_info)),
        "fraud_locations": fraud_locations,
        "_n_gt_evidence": n_gt_evidence,
    }
