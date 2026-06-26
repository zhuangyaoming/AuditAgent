"""
US market prior accounting subjects (15 categories from filtered 710 cases).

Each subject has:
  - category: standardized category name
  - keywords: English search terms for regex / BM25 / dense retrieval
  - regex: compiled regex for section_retriever exact matching
"""

import json
import re
from pathlib import Path

# ---- 15 US prior subjects (filtered 710 cases, excl. fund/investment labels) ----

US_PRIOR_15 = [
    {
        "category": "Revenue",
        "keywords": [
            "Revenue", "Revenues", "Net Sales", "Net Revenues", "Fee Revenue",
            "Interest Income", "Sales", "Service Revenue", "License Revenue",
        ],
    },
    {
        "category": "Accounts Receivable",
        "keywords": [
            "Accounts Receivable", "Trade Receivables", "Receivables",
            "Accounts Receivable, Net", "Trade Accounts Receivable",
        ],
    },
    {
        "category": "Net Income",
        "keywords": [
            "Net Income", "Net Earnings", "Net Loss", "Operating Income",
            "Pre-tax Income", "Income Before Income Taxes", "Pretax Earnings",
            "Net Profit", "Earnings",
        ],
    },
    {
        "category": "Additional Paid-in Capital",
        "keywords": [
            "Additional Paid-in Capital", "Additional Paid-In Capital",
            "Paid-in Capital", "Capital in Excess of Par",
            "Contributed Capital", "APIC",
        ],
    },
    {
        "category": "Investment Securities",
        "keywords": [
            "Investment Securities", "Equity Securities", "Marketable Securities",
            "Derivatives", "Securities Portfolio", "Debt Securities",
            "Trading Securities", "Available-for-Sale Securities",
        ],
    },
    {
        "category": "Stockholders' Equity",
        "keywords": [
            "Stockholders' Equity", "Shareholders' Equity", "Equity Capital",
            "Total Equity", "Net Equity",
        ],
    },
    {
        "category": "Common Stock",
        "keywords": [
            "Common Stock", "Capital Stock", "Share Capital",
            "Outstanding Shares", "Common Shares", "Ordinary Shares",
        ],
    },
    {
        "category": "Inventory",
        "keywords": [
            "Inventory", "Inventories", "Finished Goods",
            "Raw Materials", "Work in Progress", "Work-in-Process",
            "LIFO", "FIFO", "Inventory Reserve",
        ],
    },
    {
        "category": "Operating Expenses",
        "keywords": [
            "Operating Expenses", "Selling, General and Administrative",
            "SG&A", "General and Administrative", "Operating Costs",
        ],
    },
    {
        "category": "Compensation Expense",
        "keywords": [
            "Compensation Expense", "Stock-Based Compensation",
            "Salaries and Wages", "Share-Based Compensation",
            "Stock Option Expense", "Employee Compensation",
        ],
    },
    {
        "category": "Cost of Goods Sold",
        "keywords": [
            "Cost of Goods Sold", "Cost of Sales", "Cost of Revenues",
            "COGS", "Cost of Services",
        ],
    },
    {
        "category": "Pre-tax Income",
        "keywords": [
            "Pre-tax Income", "Income Before Income Taxes",
            "Pretax Earnings", "Pre-Tax Earnings",
            "Income Before Taxes", "Earnings Before Income Taxes",
        ],
    },
    {
        "category": "Loss Reserves",
        "keywords": [
            "Loss Reserves", "Allowance for Doubtful Accounts",
            "Loan Loss Reserves", "Allowance for Loan Losses",
            "Bad Debt Reserve", "Reserve for Doubtful Accounts",
            "Credit Loss Reserve", "Valuation Allowance",
        ],
    },
    {
        "category": "Cash and Cash Equivalents",
        "keywords": [
            "Cash and Cash Equivalents", "Cash", "Cash Equivalents",
            "Restricted Cash", "Cash on Hand",
        ],
    },
    {
        "category": "Property, Plant and Equipment",
        "keywords": [
            "Property, Plant and Equipment", "PP&E", "Fixed Assets",
            "Property and Equipment, Net", "Plant and Equipment",
            "Capital Assets", "Tangible Assets",
        ],
    },
]


def get_us_prior_15() -> list[dict]:
    """Return the 15 US market prior subjects."""
    return US_PRIOR_15


# ---- 15 CN market prior subjects (English keywords from TERM_MAPPING) ----

CN_PRIOR_15 = [
    {
        "category": "Inventory",
        "keywords": ["Inventories", "Inventory"],
    },
    {
        "category": "Accounts Receivable",
        "keywords": ["Accounts Receivable", "Trade Receivables", "Receivables"],
    },
    {
        "category": "Cash and Cash Equivalents",
        "keywords": ["Cash and Cash Equivalents", "Cash"],
    },
    {
        "category": "Other Receivables",
        "keywords": ["Other Receivables", "Other Current Assets"],
    },
    {
        "category": "Goodwill",
        "keywords": ["Goodwill"],
    },
    {
        "category": "Fixed Assets",
        "keywords": ["Property, Plant and Equipment", "PP&E", "Plant and Equipment", "Fixed Assets"],
    },
    {
        "category": "Prepaid Expenses",
        "keywords": ["Prepaid Expenses", "Prepayments"],
    },
    {
        "category": "Long-term Equity Investments",
        "keywords": ["Equity Method Investments", "Long-term Investments"],
    },
    {
        "category": "Construction in Progress",
        "keywords": ["Construction in Progress", "CIP"],
    },
    {
        "category": "Retained Earnings",
        "keywords": ["Retained Earnings", "Accumulated Deficit"],
    },
    {
        "category": "Other Non-Current Assets",
        "keywords": ["Other Non-Current Assets"],
    },
    {
        "category": "Intangible Assets",
        "keywords": ["Intangible Assets", "Acquired Technology"],
    },
    {
        "category": "Revenue and Cost of Revenue",
        "keywords": ["Revenue", "Revenues", "Net Sales", "Cost of Goods Sold", "Cost of Sales", "Cost of Revenues"],
    },
    {
        "category": "Financial Expenses",
        "keywords": ["Interest Expense", "Interest Income"],
    },
    {
        "category": "Other Payables",
        "keywords": ["Other Payables", "Other Current Liabilities", "Accrued Liabilities"],
    },
]


def get_cn_prior_15() -> list[dict]:
    """Return 15 CN market prior subjects with English keywords."""
    return CN_PRIOR_15


def get_cn_us_prior() -> list[dict]:
    """Merge CN + US prior subjects, deduplicating overlapping categories.

    CN subjects whose category or first keyword appear in any US subject
    are skipped in favor of the US version (more comprehensive keywords).
    Returns ~25 subjects.
    """
    us_categories = {s["category"].lower() for s in US_PRIOR_15}
    us_keywords_flat = set()
    for s in US_PRIOR_15:
        for kw in s["keywords"]:
            us_keywords_flat.add(kw.lower())

    merged = list(US_PRIOR_15)  # start with all US subjects
    for cn_s in CN_PRIOR_15:
        cat_lower = cn_s["category"].lower()
        first_kw = cn_s["keywords"][0].lower()
        # Check if this CN subject is already covered by US
        if cat_lower in us_categories or first_kw in us_keywords_flat:
            continue
        merged.append(cn_s)

    return merged


def load_all_categories() -> list[dict]:
    """Load all ~77 categories from account_classification.json as prior subjects (no-prior mode)."""
    classification_path = Path(
        "d:/mainfiles/AuditAgent/finfraud_processing/data/processed/cases/account_classification.json"
    )
    with open(classification_path, encoding="utf-8") as f:
        data = json.load(f)

    subjects = []
    for cat_name, cat_data in data["categories"].items():
        # Exclude the 6 fund/investment labels that were filtered out
        exclude = {
            "Investor Funds", "Client Funds", "Assets Under Management",
            "Non-Accounting Violations", "Illegal Trading Gains / Proceeds",
            "Investor Capital",
        }
        if cat_name in exclude:
            continue
        subjects.append({
            "category": cat_name,
            "keywords": cat_data["items"],
        })
    return subjects


def build_search_patterns(subjects: list[dict]) -> list[re.Pattern]:
    """Compile combined regex patterns from all subject keywords for section extraction."""
    all_keywords = []
    for s in subjects:
        for kw in s["keywords"]:
            escaped = re.escape(kw)
            all_keywords.append(escaped)
    # Sort by length descending so longer patterns match first
    all_keywords.sort(key=len, reverse=True)
    pattern_str = "|".join(all_keywords)
    return re.compile(pattern_str, re.IGNORECASE)
