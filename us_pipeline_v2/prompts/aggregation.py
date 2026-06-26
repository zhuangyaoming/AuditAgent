"""Final aggregation prompt — integrates single-report + cross-report into JSON output."""

AGGREGATION_SYSTEM = """You are a financial auditing expert. Output pure JSON only."""


def build_aggregation_prompt(
    single_results: str,
    cross_results: str,
    fraud_period_str: str = "",
) -> str:
    """Build the final aggregation prompt.

    Combines single-report analyses and cross-report synthesis,
    asks LLM to produce the final structured JSON with is_risk and risk_facts.
    """
    fp_block = ""
    if fraud_period_str:
        fp_block = (
            f"\nKnown fraud period: {fraud_period_str}\n"
        )

    return f"""# Role
You are a financial auditing expert specializing in SEC enforcement and financial fraud detection.

# Task
Integrate the single-report and cross-report risk analyses below into a final,
global fraud risk assessment for this company.

{fp_block}
# Single-Report Analyses
{single_results}

# Cross-Report Trend Analysis
{cross_results}

# Output
Output PURE JSON (no markdown, no ```json``` markers, no extra text) in this exact structure:
{{
  "is_risk": "0" or "1",
  "risk_facts": [
    {{
      "risk_title": "Short descriptive title (e.g., 'Premature Revenue Recognition on Long-Term Contracts')",
      "involved_report": "Report names, semicolon-separated (e.g., '2000年年度报告 (10-K);2001年年度报告 (10-K)')",
      "confidence": 0.0-1.0,
      "evidence_chain": [
        {{
          "point": "Brief evidence summary",
          "analysis": "Detailed analysis citing specific figures and report names from the analyses"
        }}
      ]
    }}
  ]
}}

# Requirements
1. is_risk = "0" ONLY if no fraud signals were found in ANY analysis.
2. If is_risk = "1", list up to 5 risk_facts sorted by confidence descending.
3. Each risk_fact must include specific report names in involved_report.
4. Each evidence_chain entry must cite actual figures/data from the analyses.
5. Confidence scores must reflect the strength of the evidence found.
6. OUTPUT PURE JSON ONLY — no explanation, no markdown, no ```json``` wrapper.
"""
