"""
AuditAgent 评估指标计算模块
================================
支持 LLM-as-Judge 逐条评判 + 聚合 R_I / R_E / Precision_I / F1_I。
兼容中国旧系统 label 格式和美国 FraudCase 格式。

Usage:
    from evaluation.metrics import Evaluator
    evaluator = Evaluator(model="minimax-m2.5")
    results = await evaluator.evaluate(predictions, labels)
    # results = {"R_I": 0.75, "R_E": 0.62, "Precision_I": 0.80, "F1_I": 0.774, ...}
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import aiohttp

logger = logging.getLogger(__name__)

# ============================================================
# API 配置
# ============================================================

# TODO: Set your API keys via environment variables before running.
# Original keys backed up in api_keys_backup.txt (EXCLUDED from git).
API_CONFIGS = {
    "qwq-32b": {
        "url": "https://cloud.infini-ai.com/maas/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
        "model": "qwq-32b",
    },
    "minimax-m2.5": {
        "url": "https://api.minimax.chat/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
        "model": "MiniMax-M2.5",
    },
    "minimax-m2.7-volces": {
        "url": "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
        "model": "minimax-m2.7",
    },
}


# ============================================================
# Label 格式标准化
# ============================================================

_TYPE_L2_LABELS: Dict[str, str] = {
    "F1.1": "虚构交易",
    "F1.2": "提前确认收入",
    "F1.3": "完工百分比操纵",
    "F1.4": "渠道填塞",
    "F1.5": "附条件交易",
    "F2.1": "不当费用资本化",
    "F2.2": "推迟费用确认",
    "F2.3": "准备金操纵",
    "F3.1": "存货虚增",
    "F3.2": "应收账款虚增",
    "F3.3": "资产减值隐瞒",
    "F3.4": "在建工程虚增",
    "F4.1": "表外负债隐瞒",
    "F4.2": "或有负债未披露",
    "F5.1": "关联交易未披露",
    "F5.2": "重大事项遗漏",
    "F5.3": "内控缺陷隐瞒",
    "F6.1": "腐败贿赂",
    "F6.2": "内幕交易",
    "F6.3": "FCPA违规",
    "F6.4": "其他舞弊",
}


# ASC code → matching keywords for mapping risk_title to affected_accounts indices.
# Must stay consistent with the LLM judge prompt's accounting concept matching rules.
_ASC_KEYWORD_MAP: dict[str, list[str]] = {
    "ASC 606": ["revenue recognition", "premature revenue", "bill and hold", "blra",
                 "rate collection", "revenue"],
    "ASC 360": ["capitalization", "construction", "cwip", "work in progress", "pp&e",
                "property plant", "improper asset", "asset misclassification", "ppe", "fixed asset"],
    "ASC 740": ["deferred tax", "tax position", "tax asset", "tax benefit", "income tax"],
    "ASC 450": ["reserve manipulation", "cookie jar", "excess reserve", "contingency",
                "reserve", "accrual manipulation", "accrual"],
    "ASC 350": ["impairment", "write-down", "goodwill", "intangible",
                "reserve", "asset impairment", "liability"],
    "ASC 220": ["earnings management", "income manipulation", "net income", "operating income",
                "profitability", "earnings", "income statement", "big bath",
                "special charge", "restructuring charge"],
    "ASC 230": ["cash flow", "operating cash flow"],
    "ASC 980": ["regulatory assets", "rate base", "utility"],
    "ASC 850": ["related-party", "related party"],
    "ASC 718": ["stock compensation", "equity compensation", "share-based"],
    "ASC 720": ["expense manipulation", "understated expense", "cost manipulation", "expense"],
    "ASC 330": ["inventory", "lifo", "fifo"],
    "ASC 250": ["restatement", "accounting change", "revision"],
}


def _match_risk_to_issue_indices(risk_title: str, affected_accounts: list[dict]) -> set[int]:
    """将 risk_title 文本映射到 affected_accounts 的索引集合。

    使用 ASC 编码关键字匹配，与 LLM judge 的匹配规则保持一致。
    """
    if not risk_title or not affected_accounts:
        return set()

    title_lower = risk_title.lower()
    matched: set[int] = set()

    for i, aa in enumerate(affected_accounts):
        gaap = (aa.get("gaap_codification", "") or "").lower()
        account = (aa.get("account", "") or "").lower()

        # Method 1: Direct ASC code substring match (e.g. "asc 606" in title)
        if gaap and gaap in title_lower:
            matched.add(i)
            continue

        # Method 2: Keyword group match based on mapped ASC code
        if gaap:
            keywords = _ASC_KEYWORD_MAP.get(gaap.upper(), [])
            if any(kw in title_lower for kw in keywords):
                matched.add(i)
                continue

        # Method 3: Account name keyword overlap (e.g. "revenue" from "Revenue")
        # Use a permissive matching that handles singular/plural forms.
        account_parts = re.split(r'[/,\s]+', account)
        account_keywords = [p for p in account_parts if len(p) >= 3]
        title_words = set(re.split(r'[\s,;()\[\]{}]+', title_lower))
        matched_account = False
        for kw in account_keywords:
            # Direct match: keyword is substring of title
            if kw in title_lower:
                matched_account = True
                break
            # Singular/plural: try without trailing 's'
            kw_stem = kw.rstrip('s')
            if len(kw_stem) >= 3 and kw_stem in title_lower:
                matched_account = True
                break
            # Check if any title word shares a significant prefix (>=5 chars)
            # with the account keyword
            for tw in title_words:
                if len(tw) >= 5 and len(kw) >= 5:
                    min_len = min(len(tw), len(kw))
                    if min_len >= 5 and tw[:min_len] == kw[:min_len]:
                        matched_account = True
                        break
            if matched_account:
                break
        if matched_account:
            matched.add(i)

    return matched


def _match_evidence_to_label_indices(pred_evidence_chain: list[dict], fraud_locations: list[dict]) -> set[int]:
    """将 prediction 的 evidence_chain 匹配到 GT fraud_locations 的索引集合。

    逐条 GT fraud_locations[].evidence，检查 prediction evidence_chain 中是否有相关内容匹配。
    每条 GT evidence 至多被计数一次。
    """
    matched: set[int] = set()
    if not pred_evidence_chain or not fraud_locations:
        return matched

    # 拼接 prediction evidence_chain 为文本
    pred_ev_texts = []
    for e in pred_evidence_chain:
        point = e.get("point", "") or ""
        analysis = e.get("analysis", "") or ""
        if point or analysis:
            pred_ev_texts.append(f"{point} {analysis}".lower())

    for li, loc in enumerate(fraud_locations):
        label_ev = (loc.get("evidence", "") or "").lower()
        if not label_ev:
            continue
        # 检查是否有任意一条 pred evidence 与这条 GT evidence 有足够重叠
        label_words = set(re.split(r'[\s,;()\[\]{}]+', label_ev))
        label_words = {w for w in label_words if len(w) >= 3}
        for pet in pred_ev_texts:
            pet_words = set(re.split(r'[\s,;()\[\]{}]+', pet))
            pet_words = {w for w in pet_words if len(w) >= 3}
            if label_words & pet_words:  # 有交集
                matched.add(li)
                break

    return matched


def _type_l2_label(type_l2: str) -> str:
    """将 type_l2 编码转为中文标签"""
    return _TYPE_L2_LABELS.get(type_l2, type_l2)

def chinese_label_to_unified(record: dict) -> dict:
    """将中国旧系统 Excel 行 label 转为统一格式"""
    return {
        "case_id": record.get("Symbol", record.get("case_id", "")),
        "is_fraud": int(record.get("IsFraud", 0)),
        "fraud_info": record.get("FraudInfo", []) or [],
        "fraud_locations": [
            {
                "report": loc.get("Report", ""),
                "summary": loc.get("Summary", ""),
                "evidence": loc.get("Evidence", ""),
            }
            for loc in (record.get("FraudLoc", []) or [])
        ],
    }


def _fraudcase_to_accounting_unified(case: dict) -> dict:
    """将美国 FraudCase 转为统一格式 — 使用 affected_accounts (会计问题)

    仿照旧中国系统：fraud_locations[].summary 包含会计问题描述，而非 "Issue type: F5.2"
    """
    affected_accounts = case.get("affected_accounts", [])
    fraudulent_reports = case.get("fraudulent_reports", [])
    fraud_types = case.get("fraud_types", [])

    # Build accounting issue labels from affected_accounts
    accounting_issues: List[str] = []
    for aa in affected_accounts:
        account = aa.get("account", "")
        gaap = aa.get("gaap_codification", "")
        direction = aa.get("direction", "")
        amount = aa.get("estimated_amount", "")

        label_parts = []
        if account:
            label_parts.append(account)
        if gaap:
            label_parts.append(f"({gaap})")
        if direction:
            label_parts.append(f"- {direction}")
        if amount:
            label_parts.append(f"- {amount}")

        accounting_issues.append(" ".join(label_parts))

    # Build type_l2 -> label mapping for summary enrichment
    type_l2_to_label = {}
    for ft in fraud_types:
        type_l2 = ft.get("type_l2", "")
        desc = ft.get("description", "")
        if type_l2:
            type_l2_to_label[type_l2] = {
                "cn_label": _type_l2_label(type_l2),
                "desc": desc,
            }

    # Build enriched fraud_locations (like old Chinese system's FraudLoc)
    fraud_locations: List[dict] = []
    for r in fraudulent_reports:
        report_name = _build_report_name(r)
        for issue in r.get("fraud_issues", []):
            evidence_list = issue.get("evidence", [])
            type_l2 = issue.get("type_l2", "")
            if evidence_list:
                # Build enriched summary: accounting context + fraud type label
                summary_parts = []
                if accounting_issues:
                    summary_parts.append(" | ".join(accounting_issues[:3]))
                ft_info = type_l2_to_label.get(type_l2, {})
                cn_label = ft_info.get("cn_label", type_l2)
                if cn_label:
                    summary_parts.append(f"[{type_l2} {cn_label}]")
                enriched_summary = " ".join(summary_parts)

                fraud_locations.append({
                    "report": report_name,
                    "summary": enriched_summary,
                    "evidence": "; ".join(evidence_list) if isinstance(evidence_list, list) else evidence_list,
                })

    return {
        "case_id": case.get("case_id", ""),
        "is_fraud": 1 if fraudulent_reports else 0,
        "fraud_info": accounting_issues,
        "fraud_locations": fraud_locations,
        "n_issues": len(affected_accounts),
        "affected_accounts": affected_accounts,
    }


def _fraudcase_to_coso_unified(case: dict) -> dict:
    """将美国 FraudCase 转为统一格式 — 使用 fraud_types (COSO 行为模式)"""
    fraud_types = case.get("fraud_types", {})
    fraudulent_reports = case.get("fraudulent_reports", [])

    fraud_info: List[str] = []
    fraud_locations: List[dict] = []

    # Build type_l2 to description mapping from fraud_types
    type_l2_to_desc = {}
    for ft in fraud_types:
        type_l2 = ft.get("type_l2", "")
        desc = ft.get("description", "")
        if type_l2 and desc:
            type_l2_to_desc[type_l2] = desc
            fraud_info.append(desc)

    for r in fraudulent_reports:
        report_name = _build_report_name(r)
        for issue in r.get("fraud_issues", []):
            # Use English description from fraud_types, fallback to Chinese label
            type_l2 = issue.get("type_l2", "")
            description = type_l2_to_desc.get(type_l2) or _type_l2_label(type_l2)
            evidence_list = issue.get("evidence", [])
            if evidence_list:
                fraud_locations.append({
                    "report": report_name,
                    "summary": description,
                    "evidence": "; ".join(evidence_list) if isinstance(evidence_list, list) else evidence_list,
                })

    return {
        "case_id": case.get("case_id", ""),
        "is_fraud": 1 if fraudulent_reports else 0,
        "fraud_info": fraud_info,
        "fraud_locations": fraud_locations,
        "n_issues": len(fraud_locations),
    }


def fraudcase_to_unified(case: dict, use_accounting_issues: bool = True) -> dict:
    """将美国 FraudCase (dict形式) 转为统一格式

    Args:
        use_accounting_issues: True=使用affected_accounts(会计问题), False=使用fraud_types(COSO)
    """
    affected_accounts = case.get("affected_accounts", [])
    fraud_types = case.get("fraud_types", [])

    if use_accounting_issues and affected_accounts:
        return _fraudcase_to_accounting_unified(case)
    elif use_accounting_issues and not affected_accounts and fraud_types:
        logger.warning(f"Case {case.get('case_id')} has no affected_accounts, falling back to COSO")
        return _fraudcase_to_coso_unified(case)
    else:
        return _fraudcase_to_coso_unified(case)


def _build_report_name(report: dict) -> str:
    """构造报告名称"""
    rt = report.get("report_type", "")
    fy = report.get("fiscal_year", "")
    fq = report.get("fiscal_quarter")
    name = f"{fy}年"
    report_type_cn = {"10-K": "年度报告", "10-Q": "季度报告", "Form 20-F": "年度报告 (20-F)"}
    rt_cn = report_type_cn.get(rt, rt)
    name += rt_cn
    if fq:
        name += f" {fq}"
    return name


# ============================================================
# AuditAgent 输出解析
# ============================================================

def parse_auditagent_output(raw: str) -> dict:
    """从 AuditAgent 的原始 JSON 字符串提取 risk_facts"""
    if not raw or not isinstance(raw, str):
        return {"is_risk": 0, "risk_facts": [], "market": ""}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'({[\s\S]*})', raw)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                return {"is_risk": 0, "risk_facts": [], "market": ""}
        else:
            return {"is_risk": 0, "risk_facts": [], "market": ""}
    return {
        "is_risk": data.get("is_risk", 0),
        "risk_facts": data.get("risk_facts", []),
        "market": data.get("market", ""),
    }


# ============================================================
# LLM Judge Prompt (适配自 evaluation.py)
# ============================================================

JUDGE_SYSTEM_PROMPT = """### **Role**
You are an expert evaluator of financial fraud detection models.

---

### **Task**
Assess whether a model prediction (<output>) matches any ground-truth label (<label>).

---

### **Evaluation Criteria**

1. **`risk_title`**: Is the predicted risk_title semantically consistent with ANY fraud type in the label?

   **IMPORTANT - Lenient Matching Rule**: Financial fraud can be described from different angles using different terminology. You MUST match predictions to labels if they describe the SAME underlying fraud issue, even when one uses "disclosure/omission" language and the other uses "accounting manipulation" language. These are two sides of the same coin.

   **Match as 1 (Yes) when:**
   - The prediction and label describe the same underlying fraud, regardless of terminology differences
   - A prediction about accounting manipulation (improper capitalization, asset misclassification, revenue manipulation, earnings management, cost manipulation, reserve manipulation, impairment concealment) matches a label about disclosure failures (material omissions, false/misleading statements, undisclosed information) when they relate to the SAME project, time period, or financial issue
   - A prediction about specific accounting irregularities matches a label about general fraud descriptions covering the same facts

   **Examples that MUST match (risk_title = 1):**
   - Label: "Material Omissions about project delays" <-> Prediction: "Asset Misclassification" or "Improper Capitalization" or "Cost Manipulation"
   - Label: "Undisclosed related-party transactions" <-> Prediction: "Related-Party Revenue Recognition" or "Improper Gain on Sale"
   - Label: "False/misleading statements about viability" <-> Prediction: "Earnings Management" or "Impairment Concealment"
   - Label: "Failure to disclose internal assessments" <-> Prediction: Any accounting irregularity about the same project/time period

   **Match as 0 (No) ONLY when:**
   - The prediction describes a completely unrelated financial issue
   - The prediction refers to a different company, different time period, or different project entirely

2. **`evidence_chain`**: Does the predicted evidence relate to the same category of financial issue as ANY label evidence?
   - 1 = Yes, evidence relates to the same fraud (e.g., revenue recognition, reserve manipulation, disclosure failures, asset misclassification)
   - 0 = No, evidence is about a completely different financial topic

---

### **Output Format**
Output ONLY a JSON object (no markdown, no extra text):
```json
{
  "risk_title": 0 or 1,
  "evidence_chain": 0 or 1
}
```"""


ACCOUNTING_JUDGE_SYSTEM_PROMPT = """### **Role**
You are an expert evaluator of financial fraud detection models specializing in accounting terminology.

---

### **Task**
Assess whether a model prediction (<output>) matches any ground-truth label (<label>).

---

### **Evaluation Criteria**

You are evaluating accounting-based fraud detection. The Agent predicts accounting problems (e.g., "Improper Capitalization of Construction Costs", "Revenue Recognition Manipulation"). The Ground Truth provides TWO types of information to match against:

1. **accounting_issues**: Specific affected accounts with ASC codifications (e.g., "Revenue (ASC 606) - overstated")
2. **fraud_locations[].summary**: Enriched accounting problem descriptions combining account info with fraud type context (e.g., "Assets/WIP (ASC 360) - overstated [F5.2 重大事项遗漏]")

**1. `risk_title` Matching (Primary):**
Check if the Agent's predicted `risk_title` semantically matches ANY of:
- An `accounting_issues` entry (affected account with ASC code + direction)
- A `fraud_locations[].summary` entry (accounting description with fraud context)

**Accounting Concept Matching Rules:**
Match the prediction's accounting concept to the label's accounting issue:
- "Capitalization" / "Construction" / "CWIP" / "Work in Progress" / "PP&E" / "Improper Asset" → matches **ASC 360** (Property, Plant and Equipment / Construction in Progress)
- "Revenue Recognition" / "Premature Revenue" / "Bill and Hold" / "BLRA" / "Rate Collection" → matches **ASC 606** (Revenue from Contracts with Customers)
- "Deferred Tax" / "Tax Position" / "Tax Asset" / "Tax Benefit" → matches **ASC 740** (Income Taxes)
- "Reserve Manipulation" / "Cookie Jar" / "Excess Reserve" / "Contingency" → matches **ASC 450** (Contingencies) or **ASC 350** (Impairment)
- "Inventory" / "LIFO" / "FIFO" / "Stock" → matches **ASC 330** (Inventory)
- "Earnings Management" / "Income Manipulation" / "Net Income" / "Operating Income" → matches **ASC 220** (Comprehensive Income / Net Income)
- "Asset Misclassification" / "Improper Asset Transfer" / "Asset Reclassification" → matches **ASC 360** (PP&E / Assets)
- "Regulatory Assets" / "Rate Base" / "BLRA" / "Utility" → matches **ASC 980** (Regulated Operations)
- "Related-Party" / "Related Party Revenue" → matches **ASC 850** (Related-Party Disclosures)
- "Stock Compensation" / "Equity Compensation" / "Share-Based" → matches **ASC 718** (Stock Compensation)
- "Expense Manipulation" / "Understated Expenses" / "Cost Manipulation" → matches **ASC 720** (Other Expenses)
- "Impairment Concealment" / "Asset Write-down" / "Goodwill" / "Intangible" → matches **ASC 350** (Intangibles / Impairment)
- "Cash Flow Manipulation" / "Operating Cash Flow" → matches **ASC 230** (Cash Flows)

**Match Examples:**
- Prediction: "Improper Capitalization of Construction Costs" → matches GT: "Assets (Construction in Progress) (ASC 360) - overstated" = **MATCH**
- Prediction: "Premature Revenue Recognition Through BLRA Rate Collections" → matches GT: "Revenue (Projected) (ASC 606) - overstated" = **MATCH**
- Prediction: "Nuclear Construction Project Cost Manipulation" → matches GT: "Assets/Construction in Progress (ASC 360) ... [F5.2 重大事项遗漏]" = **MATCH**

**Match as 0 (No) ONLY when:**
- The prediction describes an accounting concept completely unrelated to any accounting issue in the label
- The prediction refers to a different company, different time period, or different financial issue entirely

**2. `evidence_chain` Matching:**
Does the predicted evidence relate to the same category of financial issue as ANY label evidence?
- Check if evidence discusses the same account type (revenue, assets, expenses, etc.)
- Check if evidence references the same ASC codification or accounting concept
- 1 = Yes, evidence discusses related accounting issues
- 0 = No, evidence is about unrelated financial topics

---

### **Output Format**
Output ONLY a JSON object (no markdown, no extra text):
```json
{
  "risk_title": 0 or 1,
  "evidence_chain": 0 or 1
}
```"""


def _build_judge_user_prompt(pred_risk: dict, label: dict, match_mode: str = "accounting") -> str:
    """构造 LLM judge 的 user prompt（含上下文）

    Args:
        match_mode: "accounting" (match against affected_accounts) or "coso" (match against fraud_types)
    """
    risk_title = pred_risk.get("risk_title", "")
    involved_report = pred_risk.get("involved_report", "")
    evidence_chain = pred_risk.get("evidence_chain", [])
    evidence_text = "; ".join(
        f"{e.get('point', '')}: {e.get('analysis', '')}"
        for e in evidence_chain
    ) if isinstance(evidence_chain, list) else str(evidence_chain)

    output_obj = {
        "risk_title": risk_title,
        "involved_report": involved_report,
        "evidence_chain": evidence_text,
    }

    fraud_locations = label.get("fraud_locations", [])
    fraud_info = label.get("fraud_info", [])
    affected_accounts = label.get("affected_accounts", [])

    if match_mode == "accounting":
        label_obj = {
            "case_id": label.get("case_id", ""),
            "is_fraud": label.get("is_fraud", 0),
            "accounting_issues": fraud_info,
            "affected_accounts": affected_accounts,
            "fraud_locations": [
                {"report": loc.get("report", ""),
                 "evidence": loc.get("evidence", "")}
                for loc in fraud_locations
            ],
        }
        prompt_intro = """Extract accounting keywords from the prediction's risk_title and match them
against the GT's affected_accounts. Also check if the evidence_chain matches GT fraud_locations' evidence.

RULES:
- Accounting: Extract Revenue, COGS, Cash, Goodwill, Inventory, Accounts Receivable, Fixed Assets, etc.
- Match against GT's affected_accounts by ASC code or account name keyword overlap
- A risk_title can match multiple affected_accounts (1:N allowed)
- Only count substantive accounting concepts, not generic words like "manipulation", "fraud", "loss"

EVIDENCE MATCHING (IMPORTANT):
- You are evaluating ONE prediction against ALL GT evidence entries
- matched_evidence_count = number of GT evidence entries that THIS prediction matches (0 to total GT evidence)
- A GT evidence entry is "matched" if the prediction's evidence_chain contains related content about the same account type or fraud issue
- Each GT evidence entry can be matched by at most one prediction in the full evaluation
- matched_evidence_count MUST be <= total GT evidence count (you are counting matches, not occurrences)
- For example, if GT has 6 evidence entries and this prediction's evidence_chain covers issues related to 3 of them, return matched_evidence_count = 3

Respond JSON:
{
  "predicted_accounts": ["list of accounting concepts found in risk_title"],
  "matched_accounts": ["list of those that match GT affected_accounts"],
  "matched_count": integer,
  "total_predicted": integer,
  "matched_evidence_count": integer,
  "total_predicted_evidence": integer
}
"""
    else:
        label_obj = {
            "fraud_info": fraud_info,
            "fraud_locations": [
                {"report": loc.get("report", ""),
                 "summary": loc.get("summary", ""),
                 "evidence": loc.get("evidence", "")}
                for loc in fraud_locations
            ],
        }
        prompt_intro = """Assess whether the prediction matches any ground-truth label.

IMPORTANT: Apply LENIENT matching. If the prediction describes accounting manipulation (asset misclassification, improper capitalization, cost manipulation, earnings management, etc.) and the label describes disclosure failures (material omissions, false statements, undisclosed information) about the SAME project/time period, they MATCH. Different terminology for the same underlying fraud should be scored as risk_title=1. Do NOT penalize terminology differences.
"""

    return f"""{prompt_intro}
Note: involved_report is provided for context only. Report matching is handled separately.

<output>
{json.dumps(output_obj, ensure_ascii=False, indent=2)}
</output>

<label>
{json.dumps(label_obj, ensure_ascii=False, indent=2)}
</label>"""


# ============================================================
# API 调用
# ============================================================

async def _call_llm_judge(
    session: aiohttp.ClientSession,
    api_config: dict,
    pred_risk: dict,
    label: dict,
    sem: asyncio.Semaphore,
    match_mode: str = "accounting",
) -> dict:
    """调用 LLM judge 评判单条预测"""
    async with sem:
        user_prompt = _build_judge_user_prompt(pred_risk, label, match_mode=match_mode)
        system_prompt = ACCOUNTING_JUDGE_SYSTEM_PROMPT if match_mode == "accounting" else JUDGE_SYSTEM_PROMPT
        payload = {
            "model": api_config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        last_error = None
        for attempt in range(3):
            try:
                async with session.post(
                    api_config["url"],
                    headers=api_config["headers"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if attempt < 2:
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        logger.warning(f"LLM judge HTTP {resp.status}: {text[:200]}")
                        return {"risk_title": 0, "involved_report": 0, "evidence_chain": 0}
                    data = await resp.json()
                    content = _extract_content(data)
                    return _parse_judge_result(content)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
        logger.warning(f"LLM judge error (retries exhausted): {last_error}")
        return {"risk_title": 0, "involved_report": 0, "evidence_chain": 0}


def _extract_content(response_data) -> str:
    """从不同 API 的响应中提取文本内容"""
    if response_data is None:
        return ""
    # Check for API-level errors (quota, auth, etc.)
    base_resp = response_data.get("base_resp", {})
    if base_resp and base_resp.get("status_code", 0) != 0:
        logger.warning(f"API error: {base_resp.get('status_msg', 'unknown')}")
        return ""
    # minimax / openai format
    choices = response_data.get("choices")
    if choices and isinstance(choices, list) and len(choices) > 0:
        msg = choices[0]
        if isinstance(msg, dict):
            return msg.get("message", {}).get("content", "")
    # alternative: reply field
    if "reply" in response_data:
        return str(response_data["reply"])
    return ""


def _parse_judge_result(content: str) -> dict:
    """解析 LLM judge 的 JSON 输出 - 支持新格式（含 matched_count, total_predicted, matched_evidence_count, total_predicted_evidence）和旧格式（risk_title 0/1）"""
    # 优先尝试新格式
    for pattern in [
        r'```(?:json)?\s*\n?(.*?)\n?```',
        r'(\{[\s\S]*\})',
    ]:
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                # 新格式检测
                if "matched_count" in parsed or "total_predicted" in parsed:
                    return {
                        "matched_count": parsed.get("matched_count", 0),
                        "total_predicted": parsed.get("total_predicted", 0),
                        "matched_evidence_count": parsed.get("matched_evidence_count", 0),
                        "total_predicted_evidence": parsed.get("total_predicted_evidence", 0),
                        "risk_title": 1 if parsed.get("matched_count", 0) > 0 else 0,
                        "evidence_chain": 1 if parsed.get("matched_evidence_count", 0) > 0 else 0,
                    }
                # 旧格式检测
                if "risk_title" in parsed:
                    return parsed
        except (json.JSONDecodeError, re.error):
            continue

    # 旧格式 fallback
    if '"risk_title": 1' in content or '"evidence_chain": 1' in content:
        json_matches = re.findall(r'\{[^{}]*\}', content)
        for jm in json_matches:
            try:
                obj = json.loads(jm)
                if isinstance(obj, dict) and obj.get("risk_title") == 1:
                    return obj
                if isinstance(obj, dict) and obj.get("evidence_chain") == 1:
                    return obj
            except json.JSONDecodeError:
                continue

    return {"matched_count": 0, "total_predicted": 0, "matched_evidence_count": 0, "total_predicted_evidence": 0, "risk_title": 0, "evidence_chain": 0}


# ============================================================
# 启发式快速匹配（无需 LLM 调用，用于快速迭代）
# ============================================================

def _text_overlap_ratio(text1: str, text2: str) -> float:
    """计算两段文本的字符级重叠率"""
    if not text1 or not text2:
        return 0.0
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))


def _normalize_report_name(name: str) -> str:
    """标准化报告名称为 'YYYY 年报/季报' 形式"""
    name = name.strip()
    year_match = re.search(r'(\d{4})', name)
    year = year_match.group(1) if year_match else ""
    if "10-K" in name or "年报" in name or "年度" in name or "20-F" in name:
        rtype = "年报"
    elif "10-Q" in name or "季报" in name or "季度" in name:
        rtype = "季报"
    else:
        rtype = "报告"
    return f"{year} {rtype}"


def _parse_report_list(involved_report: str) -> list[str]:
    """将分号分隔的多报告字符串拆分为单个报告名列表"""
    if not involved_report:
        return []
    return [s.strip() for s in involved_report.split(";") if s.strip()]


def _check_reports_overlap(pred_reports: str, label_locations: list[dict], year_only: bool = False) -> bool:
    """检查预测报告是否与任意标签报告重叠（支持多报告格式）。

    year_only=True 时仅按年份匹配，不区分钟度/季度。
    """
    pred_list = _parse_report_list(pred_reports)
    if year_only:
        pred_years = set()
        for r in pred_list:
            m = re.search(r'(\d{4})', r)
            if m:
                pred_years.add(m.group(1))
        for loc in label_locations:
            m = re.search(r'(\d{4})', loc.get("report", ""))
            if m and m.group(1) in pred_years:
                return True
        return False

    pred_norms = {_normalize_report_name(r) for r in pred_list}
    for loc in label_locations:
        label_norm = _normalize_report_name(loc.get("report", ""))
        if label_norm in pred_norms:
            return True
    return False


def split_predictions(risk_facts: list[dict]) -> tuple[list[dict], list[int]]:
    """将跨报告预测按 involved_report 中的 ';' 拆分为逐报告预测。

    Returns:
        split_facts: 拆分后的 risk_facts（每个只含单份报告名）
        parent_idx: 每个拆分项对应的原始预测索引
    """
    split_facts: list[dict] = []
    parent_idx: list[int] = []
    for i, rf in enumerate(risk_facts):
        involved = rf.get("involved_report", "")
        reports = _parse_report_list(involved)
        if not reports:
            split_facts.append(rf)
            parent_idx.append(i)
        else:
            for r in reports:
                split_rf = dict(rf)
                split_rf["involved_report"] = r
                split_facts.append(split_rf)
                parent_idx.append(i)
    return split_facts, parent_idx


def heuristic_match(pred_risk: dict, label: dict, threshold: float = 0.3) -> dict:
    """
    启发式匹配：不调用 LLM，用文本重叠率判断。
    用于快速调试和离线测试。
    """
    risk_title = pred_risk.get("risk_title", "")
    involved_report = pred_risk.get("involved_report", "")

    # 检查 risk_title 是否与 label 的任何 fraud_info / summary 匹配
    summaries = [loc.get("summary", "") for loc in label.get("fraud_locations", [])]
    summaries.extend(label.get("fraud_info", []))
    title_match = 0
    for s in summaries:
        if s and _text_overlap_ratio(risk_title, s) >= threshold:
            title_match = 1
            break

    # 检查 involved_report 匹配（支持多报告引用）
    report_match = 1 if _check_reports_overlap(involved_report, label.get("fraud_locations", [])) else 0

    # 检查 evidence_chain 匹配
    evidence_chain = pred_risk.get("evidence_chain", [])
    pred_evidence_text = "; ".join(
        f"{e.get('point', '')} {e.get('analysis', '')}"
        for e in evidence_chain
    ) if isinstance(evidence_chain, list) else str(evidence_chain)

    evidence_match = 0
    for loc in label.get("fraud_locations", []):
        label_evidence = loc.get("evidence", "")
        if label_evidence and _text_overlap_ratio(pred_evidence_text, label_evidence) >= threshold:
            evidence_match = 1
            break

    return {
        "risk_title": title_match,
        "involved_report": report_match,
        "evidence_chain": evidence_match,
    }


# ============================================================
# 核心：评估器
# ============================================================

@dataclass
class CaseMetrics:
    """单个案例的指标 — 双层级 TP:
    - tp_issue_theme / n_label_issues → R_I (主题级，原始预测命中标签)
    - tp_issue_theme / n_pred_themes → Precision_I (主题级，原始预测命中)
    """
    case_id: str = ""
    n_label_issues: int = 0
    n_label_evidence: int = 0
    n_pred_risks: int = 0        # 拆分后的逐报告预测数
    n_pred_themes: int = 0       # 原始预测主题数（Precision 分母）
    tp_issue: int = 0            # 逐报告级 TP
    tp_issue_theme: int = 0      # 主题级 TP（R_I 和 Precision 分子）
    tp_evidence: int = 0         # Evidence True Positives（R_E 分子）
    matched_gt_issue_count: int = 0  # 命中的 affected_accounts 种类数（用于 R_I 和 Precision 分子）
    n_pred_accounts: int = 0         # Agent 预测的会计科目总数（Precision 分母）
    n_pred_evidence: int = 0          # Agent 预测的证据总数（R_E 分母）

    # 旧系统风格 Boolean OR 召回 — 每个 case 只要任一预测命中该维度即为 True
    is_risk_match: bool = False       # pred.is_risk 与 label.is_fraud 匹配
    any_title_match: bool = False     # 任一预测 risk_title 命中
    any_report_match: bool = False    # 任一预测 involved_report 命中
    any_evidence_match: bool = False  # 任一预测 evidence_chain 命中

    @property
    def fn_issue(self) -> int:
        return max(0, self.n_label_issues - self.matched_gt_issue_count)

    @property
    def fp_issue(self) -> int:
        return max(0, self.n_pred_themes - self.tp_issue_theme)

    @property
    def r_i(self) -> float:
        """主题级召回率：有多少 affected_accounts 种类被至少一个预测命中"""
        if self.n_label_issues == 0:
            return 0.0
        return self.matched_gt_issue_count / self.n_label_issues

    @property
    def precision_i(self) -> float:
        """主题级精确率：原始预测中有多少主题是有效的"""
        if self.n_pred_themes == 0:
            return 0.0
        return self.tp_issue_theme / self.n_pred_themes

    @property
    def f1_i(self) -> float:
        p, r = self.precision_i, self.r_i
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def r_e(self) -> float:
        if self.n_label_evidence == 0:
            return 0.0
        return self.tp_evidence / self.n_label_evidence


@dataclass
class EvalResult:
    """评估汇总结果 — 双层级"""
    R_I: float = 0.0
    R_E: float = 0.0
    Precision_I: float = 0.0
    Precision_E: float = 0.0  # 证据精确率
    F1_I: float = 0.0
    n_cases: int = 0
    total_label_issues: int = 0
    total_pred_risks: int = 0       # 拆分后逐报告预测总数
    total_pred_themes: int = 0      # 原始预测主题总数
    total_tp_issues: int = 0        # 逐报告级 TP
    total_tp_theme: int = 0         # 主题级 TP
    total_tp_evidence: int = 0
    total_label_evidence: int = 0
    per_case: List[CaseMetrics] = field(default_factory=list)
    match_mode: str = "heuristic"

    # 旧系统风格 Boolean OR 召回率（per-case 任一预测命中即为 1）
    R_is_risk: float = 0.0       # is_risk 匹配率
    R_title_bool: float = 0.0    # risk_title 任一命中率
    R_report_bool: float = 0.0   # involved_report 任一命中率
    R_evidence_bool: float = 0.0 # evidence_chain 任一命中率
    R_overall: float = 0.0       # 4 维全部命中的完全匹配率


class Evaluator:
    """评估器：judge + aggregate 一体化"""

    def __init__(
        self,
        model: str = "minimax-m2.5",
        match_mode: str = "llm",  # "llm" | "heuristic"
        evaluation_mode: str = "accounting",  # "accounting" | "coso"
        concurrency: int = 5,
    ):
        if match_mode == "llm" and model not in API_CONFIGS:
            raise ValueError(f"未知模型 '{model}'，可用: {list(API_CONFIGS)}")
        self.model = model
        self.api_config = API_CONFIGS.get(model, {})
        self.match_mode = match_mode
        self.evaluation_mode = evaluation_mode
        self.concurrency = concurrency

    def _match_evidence_simple(self, pred_risk: dict, fraud_locations: list, already_matched: set = None) -> int:
        """简单的 evidence 匹配：使用文本重叠率

        Args:
            pred_risk: 单个预测（可能是 split_fact）
            fraud_locations: GT 的 fraud_locations 列表
            already_matched: 已经被匹配的 GT evidence 索引集合（用于去重）

        Returns:
            本次新匹配的 GT evidence 条数
        """
        if already_matched is None:
            already_matched = set()

        # 提取 prediction 的 evidence_chain 内容
        ev_chain = pred_risk.get("evidence_chain", [])
        pred_ev_text = ""
        if isinstance(ev_chain, list):
            pred_ev_text = "; ".join(
                e.get("point", "") + " " + e.get("analysis", "")
                for e in ev_chain
            )
        else:
            pred_ev_text = str(ev_chain)

        if not pred_ev_text.strip():
            return 0

        # 对每个 GT evidence，检查是否有文本重叠（跳过已匹配的）
        newly_matched = 0
        threshold = 0.1  # 较低的阈值，更容易匹配

        for idx, loc in enumerate(fraud_locations):
            if idx in already_matched:
                continue  # 跳过已匹配的 evidence

            label_evidence = loc.get("evidence", "")
            if not label_evidence.strip():
                continue

            overlap = _text_overlap_ratio(pred_ev_text, label_evidence)
            if overlap >= threshold:
                newly_matched += 1
                already_matched.add(idx)  # 标记为已匹配

        return newly_matched

    async def evaluate(
        self,
        predictions: List[dict],
        labels: List[dict],
    ) -> EvalResult:
        """
        批量评估。

        Args:
            predictions: AuditAgent 输出列表，每条是 {"is_risk", "risk_facts": [...]}
            labels: 统一格式 label 列表
        Returns:
            EvalResult with aggregated metrics
        """
        if len(predictions) != len(labels):
            raise ValueError(f"预测数量({len(predictions)})与标签数量({len(labels)})不一致")

        sem = asyncio.Semaphore(self.concurrency)
        per_case: List[CaseMetrics] = []

        async with aiohttp.ClientSession() as session:
            for pred, label in zip(predictions, labels):
                case_metrics = await self._evaluate_single(session, sem, pred, label)
                per_case.append(case_metrics)

        return self._aggregate(per_case)

    async def _evaluate_single(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        pred: dict,
        label: dict,
    ) -> CaseMetrics:
        """评估单个案例 — 双层级 TP 计数"""
        risk_facts = pred.get("risk_facts", [])
        fraud_locations = label.get("fraud_locations", [])
        fraud_info = label.get("fraud_info", [])

        n_original_themes = len(risk_facts)

        # 拆分为逐报告预测
        split_facts, parent_idx = split_predictions(risk_facts)

        # Determine n_label_issues based on evaluation_mode
        if self.evaluation_mode == "accounting":
            n_label_issues = len(label.get("affected_accounts", []))
        else:
            n_label_issues = label.get("n_issues", len(fraud_locations))

        cm = CaseMetrics(
            case_id=label.get("case_id", ""),
            n_label_issues=n_label_issues,
            n_label_evidence=sum(1 for loc in fraud_locations if loc.get("evidence")),
            n_pred_risks=len(split_facts),
            n_pred_themes=n_original_themes,
        )

        # 旧系统风格 is_risk 匹配（程序化比较，无需 LLM judge）
        pred_is_risk = str(pred.get("is_risk", "0"))
        label_is_fraud = str(label.get("is_fraud", "0"))
        cm.is_risk_match = (pred_is_risk == "1" and label_is_fraud == "1")

        if not split_facts:
            return cm

        # 对拆分后的每条预测进行 judge
        if self.match_mode == "heuristic":
            match_func = lambda rf: heuristic_match(rf, label)
            tasks = None
        else:
            match_func = None
            tasks = [
                _call_llm_judge(session, self.api_config, rf, label, sem, match_mode=self.evaluation_mode)
                for rf in split_facts
            ]

        if tasks:
            judge_results = await asyncio.gather(*tasks)
        else:
            judge_results = [match_func(rf) for rf in split_facts]

        # 统一 accounting-based 匹配：直接使用 LLM 输出的 matched_count / total_predicted
        # + evidence matching + Boolean OR 召回
        theme_hit: set[int] = set()
        matched_gt_evidence_indices: set[int] = set()  # 仅 COSO mode 使用
        matched_gt_issues: set[int] = set()
        any_report_ok = False
        total_predicted_accounts = 0
        total_matched_evidence = 0  # R_E 分子（matched_evidence_count 累加）
        total_predicted_evidence = 0  # R_E 分母（原始 prediction 的 evidence_chain 条数）

        # 记录已处理的 parent_idx，避免重复累加 total_predicted_evidence
        processed_parent_indices: set[int] = set()
        # 记录已匹配的 GT evidence 索引，避免重复计数
        matched_evidence_indices: set[int] = set()

        for i, jr in enumerate(judge_results):
            title_ok = jr.get("risk_title", 0) == 1
            if self.match_mode == "heuristic":
                report_ok = jr.get("involved_report", 0) == 1
            else:
                report_ok = _check_reports_overlap(
                    split_facts[i].get("involved_report", ""), fraud_locations,
                    year_only=(self.evaluation_mode == "accounting")
                )
            evidence_ok = jr.get("evidence_chain", 0) == 1

            # Accounting mode: 直接用 LLM 输出的 matched_count / total_predicted
            # + evidence matching + Boolean OR 召回
            if self.evaluation_mode == "accounting":
                mc = jr.get("matched_count", 0)
                tp = jr.get("total_predicted", 0)
                total_predicted_accounts += tp
                # 不使用 LLM judge 返回的 matched_evidence_count，因为 LLM 可能误解 prompt
                # 改用简单的文本重叠率匹配
                total_matched_evidence += self._match_evidence_simple(split_facts[i], fraud_locations, matched_evidence_indices)
                # 手动计算 total_predicted_evidence: 只对原始 risk_facts 累加一次
                parent_i = parent_idx[i]
                if parent_i not in processed_parent_indices:
                    processed_parent_indices.add(parent_i)
                    # 获取原始 risk_fact 的 evidence_chain 条数
                    orig_rf = risk_facts[parent_i]
                    ev_chain = orig_rf.get("evidence_chain", [])
                    if isinstance(ev_chain, list):
                        total_predicted_evidence += len(ev_chain)
                    else:
                        total_predicted_evidence += 1  # 字符串形式视为 1 条

                if mc > 0 and report_ok:
                    theme_hit.add(parent_idx[i])
                    # 记录命中的 GT indices（用于 Recall 分子去重）
                    risk_title = split_facts[i].get("risk_title", "")
                    issue_indices = _match_risk_to_issue_indices(risk_title, label.get("affected_accounts", []))
                    matched_gt_issues.update(issue_indices)
            else:
                # COSO mode: 保持原有逻辑
                if title_ok and report_ok:
                    theme_hit.add(parent_idx[i])
                    for li, loc in enumerate(fraud_locations):
                        pred_report = split_facts[i].get("involved_report", "")
                        label_report = loc.get("report", "")
                        if pred_report and label_report:
                            norm_pred = _normalize_report_name(pred_report)
                            norm_label = _normalize_report_name(label_report)
                            if norm_label in norm_pred:
                                matched_gt_issues.add(li)

            if report_ok:
                any_report_ok = True

            if title_ok and report_ok:
                cm.tp_issue += 1

            # COSO evidence matching（accounting mode 不走这里）
            if self.evaluation_mode != "accounting" and evidence_ok and title_ok and report_ok:
                for li, loc in enumerate(fraud_locations):
                    if loc.get("evidence"):
                        pred_report = split_facts[i].get("involved_report", "")
                        label_report = loc.get("report", "")
                        if pred_report and label_report:
                            norm_pred = _normalize_report_name(pred_report)
                            norm_label = _normalize_report_name(label_report)
                            if norm_label in norm_pred:
                                matched_gt_evidence_indices.add(li)

        cm.tp_issue_theme = len(theme_hit)
        cm.matched_gt_issue_count = len(matched_gt_issues)  # unique GT account indices, not accumulated raw matched counts
        cm.n_pred_accounts = total_predicted_accounts       # Precision 分母
        cm.tp_evidence = total_matched_evidence             # R_E 分子（matched_evidence_count 累加）
        cm.n_pred_evidence = total_predicted_evidence       # R_E 分母（total_predicted_evidence 累加）

        # Boolean OR 召回
        cm.any_title_match = len(theme_hit) > 0
        cm.any_report_match = any_report_ok
        cm.any_evidence_match = total_matched_evidence > 0

        return cm

    def _aggregate(self, per_case: List[CaseMetrics]) -> EvalResult:
        """聚合所有案例指标为整体指标"""
        total_tp = sum(c.tp_issue for c in per_case)
        total_label = sum(c.n_label_issues for c in per_case)
        total_pred = sum(c.n_pred_risks for c in per_case)
        total_themes = sum(c.n_pred_themes for c in per_case)
        total_tp_theme = sum(c.tp_issue_theme for c in per_case)
        total_tp_ev = sum(c.tp_evidence for c in per_case)
        total_label_ev = sum(c.n_label_evidence for c in per_case)

        result = EvalResult(
            n_cases=len(per_case),
            total_label_issues=total_label,
            total_pred_risks=total_pred,
            total_pred_themes=total_themes,
            total_tp_issues=total_tp,
            total_tp_theme=total_tp_theme,
            total_tp_evidence=total_tp_ev,
            total_label_evidence=total_label_ev,
            per_case=per_case,
            match_mode=self.match_mode,
        )

        if total_label > 0:
            result.R_I = sum(c.matched_gt_issue_count for c in per_case) / total_label
        total_pred_accounts = sum(c.n_pred_accounts for c in per_case)
        if total_pred_accounts > 0:
            result.Precision_I = sum(c.matched_gt_issue_count for c in per_case) / total_pred_accounts
        p, r = result.Precision_I, result.R_I
        if p + r > 0:
            result.F1_I = 2 * p * r / (p + r)
        # Evidence metrics: 直接用 LLM 返回的 matched_evidence_count / total_predicted_evidence
        total_matched_ev = sum(c.tp_evidence for c in per_case)      # matched_evidence_count 累加
        total_pred_ev = sum(c.n_pred_evidence for c in per_case)    # total_predicted_evidence 累加
        if total_pred_ev > 0:
            result.Precision_E = total_matched_ev / total_pred_ev
            result.R_E = total_matched_ev / total_pred_ev  # R_E 直接用 LLM 返回的两个值相除

        # 旧系统风格 Boolean OR 召回率
        n = len(per_case)
        if n > 0:
            result.R_is_risk = sum(1 for c in per_case if c.is_risk_match) / n
            result.R_title_bool = sum(1 for c in per_case if c.any_title_match) / n
            result.R_report_bool = sum(1 for c in per_case if c.any_report_match) / n
            result.R_evidence_bool = sum(1 for c in per_case if c.any_evidence_match) / n
            result.R_overall = sum(1 for c in per_case
                                   if c.is_risk_match and c.any_title_match
                                   and c.any_report_match and c.any_evidence_match) / n

        return result


# ============================================================
# 便捷函数
# ============================================================

def print_eval_result(result: EvalResult) -> None:
    """格式化打印评估结果（双层级）"""
    print("\n" + "=" * 60)
    print("评估结果 (双层级)")
    print("=" * 60)
    print(f"匹配模式:   {result.match_mode}")
    print(f"案例数:     {result.n_cases}")
    print(f"标签问题数: {result.total_label_issues}")
    print(f"标签证据数: {result.total_label_evidence}")
    print(f"预测拆分数: {result.total_pred_risks} (逐报告)")
    print(f"预测主题数: {result.total_pred_themes} (原始)")
    print(f"TP 逐报告:  {result.total_tp_issues}")
    print(f"TP 主题级:  {result.total_tp_theme}")
    print(f"TP (证据):  {result.total_tp_evidence}")
    print("-" * 60)
    print(f"R_I (Issue Recall, report-level):       {result.R_I:.4f}")
    print(f"R_E (Evidence Recall):                  {result.R_E:.4f}")
    print(f"Precision_I (Issue Prec, theme-level):  {result.Precision_I:.4f}")
    print(f"F1_I (Harmonic mean of theme-P & report-R): {result.F1_I:.4f}")
    print("=" * 60)


# ============================================================
# 自测试：用 Nortel 案例验证
# ============================================================

async def _quick_test():
    """快速测试：Nortel Networks 案例"""
    import sys
    sys.path.insert(0, "..")

    # 模拟 prediction（从 us_adapter 输出中提取）
    pred = {
        "is_risk": 1,
        "risk_facts": [
            {
                "risk_title": "虚增收入",
                "involved_report": "2000年年度报告 (10-K)",
                "evidence_chain": [
                    {"point": "提前确认收入", "analysis": "收入在发货前即被确认"},
                ],
            },
            {
                "risk_title": "不当准备金转回",
                "involved_report": "2003年年度报告 (10-K)",
                "evidence_chain": [
                    {"point": "超额准备金被转回", "analysis": "2002年计提的过剩准备金在2003年被转回"},
                ],
            },
        ],
    }

    # 模拟 label（从 FraudCase 转换）
    label = fraudcase_to_unified({
        "case_id": "LR-20333",
        "fraudulent_reports": [
            {
                "report_type": "10-K", "fiscal_year": 2000,
                "fraud_issues": [
                    {"type_l2": "F1.1", "description": "虚构收入",
                     "evidence": ["Revenue from undisclosed side agreements"]},
                    {"type_l2": "F1.2", "description": "提前确认收入",
                     "evidence": ["Revenue recognized before delivery"]},
                ],
            },
            {
                "report_type": "10-K", "fiscal_year": 2003,
                "fraud_issues": [
                    {"type_l2": "F2.3", "description": "不当准备金转回",
                     "evidence": ["Excess reserves released to boost earnings"]},
                ],
            },
        ],
    })

    print("测试：启发式快速匹配")
    evaluator = Evaluator(match_mode="heuristic")
    result = await evaluator.evaluate([pred], [label])
    print_eval_result(result)

    for cm in result.per_case:
        print(f"\n  {cm.case_id}: TP={cm.tp_issue}/{cm.n_label_issues} issues, "
              f"R_I={cm.r_i:.3f}, Prec={cm.precision_i:.3f}")


if __name__ == "__main__":
    asyncio.run(_quick_test())
