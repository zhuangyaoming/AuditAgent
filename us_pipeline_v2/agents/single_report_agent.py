"""Single-report analysis agent — analyzes retrieved subject passages per report."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import aiohttp

from agents.base_agent import BaseAgent, truncate_by_tokens
from prompts.single_report import build_single_report_prompt


class SingleReportAgent(BaseAgent):
    """Analyzes one report's retrieved accounting-subject passages for fraud signals."""

    def __init__(
        self,
        model_name: str = "minimax-m2.5",
        max_retries: int = 3,
        max_tokens: int = 30000,
    ):
        super().__init__(model_name=model_name, max_retries=max_retries)
        self.max_tokens = max_tokens

    async def analyze(
        self,
        report_name: str,
        retrieved_text: str,
        fraud_period_str: str = "",
        session: aiohttp.ClientSession | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> str:
        """Run single-report fraud analysis on retrieved text.

        Args:
            report_name: Report identifier (e.g. "2000年年度报告 (10-K)").
            retrieved_text: Fused text from multi-path retrieval for this report.
            fraud_period_str: Optional known fraud period hint.

        Returns:
            Raw LLM analysis output text.
        """
        prompt = build_single_report_prompt(
            report_name=report_name,
            retrieved_text=retrieved_text,
            fraud_period_str=fraud_period_str,
        )
        prompt = truncate_by_tokens(prompt, 60000 - 5000)
        return await self.chat(prompt, session, sem, track_phase="single_report")


async def run_single_report_analyses(
    report_retrieved: list[dict],
    fraud_period_str: str = "",
    model_name: str = "minimax-m2.5",
    session: aiohttp.ClientSession | None = None,
    sem: asyncio.Semaphore | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Run single-report analysis for all reports in parallel.

    Returns:
        (list of {report_name, output} dicts, token_usage dict).
    """
    agent = SingleReportAgent(model_name=model_name)

    async def _one(report: dict) -> dict:
        output = await agent.analyze(
            report_name=report["report_name"],
            retrieved_text=report["retrieved_text"],
            fraud_period_str=fraud_period_str,
            session=session,
            sem=sem,
        )
        return {"report_name": report["report_name"], "output": output}

    tasks = [asyncio.create_task(_one(r)) for r in report_retrieved]
    results = await asyncio.gather(*tasks)
    return list(results), dict(agent.token_usage)
