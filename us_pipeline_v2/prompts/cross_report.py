"""Cross-report aggregation prompt template."""

CROSS_REPORT_SYSTEM = """You are a financial auditing expert specializing in synthesizing cross-period analyses."""


def build_cross_report_prompt(
    trend_results: str,
    fraud_period_str: str = "",
) -> str:
    """Build the cross-report aggregation prompt.

    Merges outputs from all term-trend analyses and asks the LLM
    to identify consistent cross-report risk patterns.
    """
    fp_block = ""
    if fraud_period_str:
        fp_block = (
            f"\nThe known fraud period is: {fraud_period_str}. "
            f"Cross-reference anomalies with this window.\n"
        )

    return f"""# Role
You are a financial auditing expert specializing in synthesizing cross-period analyses.

# Task
Review the aggregated cross-report trend analyses below for multiple accounting subjects.
Identify consistent patterns, corroborating signals, or contradictions across different
subjects that collectively indicate financial reporting fraud.

{fp_block}
# Aggregated Cross-Report Trend Analyses
{trend_results}

# Output Format
Output structured text (NOT JSON) covering:
1. Cross-subject patterns: which subjects show consistent anomalies?
2. Corroborating signals: do multiple subjects point to the same type of fraud?
3. Contradictory signals: are there subjects that appear normal while others are abnormal?
4. Overall cross-report risk assessment with specific examples.

# Requirements
1. Reference specific subjects and report periods.
2. Focus on patterns that span multiple subjects (stronger signal).
3. Do not fabricate data — base conclusions on the provided analyses.
"""
