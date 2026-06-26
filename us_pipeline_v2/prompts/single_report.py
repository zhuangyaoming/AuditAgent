"""Single-report analysis prompt template (English, US market)."""

SINGLE_REPORT_SYSTEM = """You are a financial auditing expert specializing in US GAAP and SEC enforcement patterns."""


def build_single_report_prompt(
    report_name: str,
    retrieved_text: str,
    fraud_period_str: str = "",
) -> str:
    """Build the single-report analysis prompt.

    The LLM receives *retrieved* text (not full report) — key accounting
    subject passages extracted via multi-path retrieval from the original filing.
    """
    fp_block = ""
    if fraud_period_str:
        fp_block = (
            f"\nThe known fraud period is: {fraud_period_str}. "
            f"Pay special attention to transactions and disclosures within this window.\n"
        )

    return f"""# Role
You are a financial auditing expert specializing in US GAAP and SEC enforcement patterns.

# Task
Analyze the extracted key accounting subject passages from a US public company's SEC filing
({report_name}) to identify potential financial reporting fraud indicators.

{fp_block}
# Context
The text below was retrieved from the full filing by searching for the most relevant
accounting subjects commonly associated with financial fraud (e.g., Revenue, Accounts
Receivable, Inventory, Reserves, etc.). Focus your analysis on these extracted passages.

# Key Fraud Indicators to check
1. Revenue Recognition: premature recognition, channel stuffing, round-tripping, fictitious sales
2. Expense Manipulation: improper capitalization, delayed expense recognition, cookie-jar reserves
3. Asset Overstatement: inflated inventory, overstated receivables, impaired assets not written down
4. Liability Understatement: off-balance-sheet obligations, undisclosed contingencies
5. Disclosure Violations: related-party transactions, missing segment info, misleading non-GAAP metrics

# Extracted Report Text
{retrieved_text}

# Output Format
Output structured text (NOT JSON) covering:
1. Overall risk assessment (0 or 1)
2. For each identified risk signal:
   - risk_title: short descriptive title
   - involved_report: "{report_name}"
   - confidence: 0.0-1.0
   - evidence_chain: list of specific evidence points with actual figures/data from the text

# Requirements
1. Base conclusions ONLY on the provided text — do not fabricate data.
2. If no risk signals are present, state that clearly.
3. Cite specific dollar amounts, ratios, or trends from the text.
4. List up to 5 distinct risk signals, sorted by confidence descending.
"""
