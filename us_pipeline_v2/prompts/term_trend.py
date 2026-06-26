"""Term trend (cross-report) analysis prompt template."""

TERM_TREND_SYSTEM = """You are a financial auditing expert specializing in cross-period accounting analysis."""


def build_term_trend_prompt(
    term_name: str,
    trend_content: str,
    fraud_period_str: str = "",
) -> str:
    """Build the term-level cross-report trend analysis prompt.

    Called once per accounting subject, concatenating that subject's passages
    across all available reports for longitudinal analysis.
    """
    fp_block = ""
    if fraud_period_str:
        fp_block = (
            f"\nThe known fraud period is: {fraud_period_str}. "
            f"Watch for unusual fluctuations or anomalies during this window.\n"
        )

    return f"""# Role
You are a financial auditing expert specializing in cross-period accounting analysis.

# Task
Analyze the trend of the accounting subject **{term_name}** across multiple reporting periods
from a US public company. Look for anomalous fluctuations, unusual patterns, or indicators
of financial manipulation.

{fp_block}
# Cross-Report Data for {term_name}
{trend_content}

# Output Format
Output structured text covering:
1. Summary of the trend for {term_name} across the reports
2. Key metrics and their period-over-period changes (cite actual figures)
3. Any anomalies or red flags (unusual spikes, drops, ratio changes, or inconsistencies)
4. Specific risk signals if any are identified

# Requirements
1. Cite specific figures and report names in your analysis.
2. Compare year-over-year or quarter-over-quarter changes where data permits.
3. Flag any accounting estimate changes, policy changes, or unusual disclosures.
4. Be concise — focus on material changes and anomalies.
"""
