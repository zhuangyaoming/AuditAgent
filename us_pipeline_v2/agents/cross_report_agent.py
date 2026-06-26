"""Cross-report trend analysis agent — pivots accounting subjects across reports."""

import asyncio
from collections import defaultdict

import aiohttp

from agents.base_agent import BaseAgent, truncate_by_tokens
from prompts.term_trend import build_term_trend_prompt


class CrossReportAgent(BaseAgent):
    """Analyzes one accounting subject's trend across all reports."""

    def __init__(
        self,
        model_name: str = "minimax-m2.5",
        max_retries: int = 3,
    ):
        super().__init__(model_name=model_name, max_retries=max_retries)

    async def analyze_trend(
        self,
        term_name: str,
        trend_content: str,
        fraud_period_str: str = "",
        session: aiohttp.ClientSession | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> str:
        """Run cross-report trend analysis for one accounting subject.

        Args:
            term_name: Accounting subject name (e.g. "Revenue").
            trend_content: Concatenated passages for this subject across all reports.
            fraud_period_str: Optional fraud period hint.

        Returns:
            Raw LLM trend analysis output.
        """
        prompt = build_term_trend_prompt(
            term_name=term_name,
            trend_content=trend_content,
            fraud_period_str=fraud_period_str,
        )
        prompt = truncate_by_tokens(prompt, 60000 - 5000)
        return await self.chat(prompt, session, sem, track_phase="cross_report")


def pivot_by_subject(report_sections: dict[str, dict[str, str]]) -> dict[str, str]:
    """Pivot per-report subject sections into per-subject cross-report content.

    Args:
        report_sections: {report_name: {subject: text, ...}, ...}
                         Each subject's text is the fused retrieval for that subject.

    Returns:
        {subject_name: concatenated_text_across_reports}
    """
    subject_data: dict[str, list[str]] = defaultdict(list)
    for report_name, subjects in report_sections.items():
        for subject, text in subjects.items():
            subject_data[subject].append(f"=== {report_name} ===\n{text}")

    return {
        subject: "\n\n".join(entries)
        for subject, entries in subject_data.items()
    }


async def run_cross_report_analyses(
    subject_content: dict[str, str],
    fraud_period_str: str = "",
    model_name: str = "minimax-m2.5",
    session: aiohttp.ClientSession | None = None,
    sem: asyncio.Semaphore | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Run cross-report trend analysis for all subjects in parallel.

    Returns:
        (list of {term, output} dicts, token_usage dict).
    """
    agent = CrossReportAgent(model_name=model_name)

    async def _one(term: str, content: str) -> dict:
        output = await agent.analyze_trend(
            term_name=term,
            trend_content=content,
            fraud_period_str=fraud_period_str,
            session=session,
            sem=sem,
        )
        return {"term": term, "output": output}

    tasks = [
        asyncio.create_task(_one(term, content))
        for term, content in subject_content.items()
    ]
    results = await asyncio.gather(*tasks)
    return list(results), dict(agent.token_usage)
