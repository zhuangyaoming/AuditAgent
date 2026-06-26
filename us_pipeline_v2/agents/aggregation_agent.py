"""Aggregation agent — integrates single + cross report results into final JSON."""

import asyncio
import json
import logging

import aiohttp

from agents.base_agent import BaseAgent, truncate_by_tokens
from prompts.cross_report import build_cross_report_prompt
from prompts.aggregation import build_aggregation_prompt


class AggregationAgent(BaseAgent):
    """Two-step aggregation: cross-report synthesis → final JSON."""

    def __init__(
        self,
        model_name: str = "minimax-m2.5",
        max_retries: int = 3,
        max_json_retries: int = 5,
    ):
        super().__init__(model_name=model_name, max_retries=max_retries)
        self.max_json_retries = max_json_retries

    async def aggregate(
        self,
        single_results: list[dict],
        cross_results: list[dict],
        fraud_period_str: str = "",
        session: aiohttp.ClientSession | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> str:
        """Run the two-step aggregation pipeline.

        Step 1: Cross-report synthesis from trend analyses.
        Step 2: Final JSON integration of single + cross results.

        Args:
            single_results: [{report_name, output}, ...]
            cross_results: [{term, output}, ...]
            fraud_period_str: Optional fraud period hint.

        Returns:
            Final JSON string (parsed and re-serialized).
        """
        merge_single = "\n\n".join(
            f"{r['report_name']}--\n{r['output']}"
            for r in single_results if r.get("output")
        )
        merge_cross = "\n\n".join(
            f"{r['term']}--\n{r['output']}"
            for r in cross_results if r.get("output")
        )

        # Step 1: Cross-report synthesis
        cross_prompt = build_cross_report_prompt(merge_cross, fraud_period_str)
        cross_prompt = truncate_by_tokens(cross_prompt, 60000)
        cross_synthesis = await self.chat(cross_prompt, session, sem, track_phase="aggregation")
        logging.info("Cross-report synthesis complete")

        # Step 2: Final JSON aggregation with retry (temperature=0 for valid JSON)
        last_output = ""
        for attempt in range(self.max_json_retries):
            aggre_prompt = build_aggregation_prompt(
                merge_single, cross_synthesis, fraud_period_str
            )
            aggre_prompt = truncate_by_tokens(aggre_prompt, 60000)
            aggre_output = await self.chat(
                aggre_prompt, session, sem, temperature=0.1,
                track_phase="aggregation",
            )
            last_output = aggre_output[:300] if aggre_output else "(empty)"

            parsed = self.parse_json_response(aggre_output)
            if parsed:
                normalized = self._normalize_aggregate_output(parsed)
                if normalized.get("risk_facts") or normalized.get("is_risk") == "0":
                    normalized["market"] = "US"
                    return json.dumps(normalized, ensure_ascii=False, indent=2)
            logging.warning(
                f"Aggregation JSON parse attempt {attempt + 1}/{self.max_json_retries} failed. "
                f"Raw output: {last_output}"
            )

        # Fallback after all retries
        return json.dumps({
            "error": "JSON parse failed after retries",
            "is_risk": "0",
            "risk_facts": [],
            "market": "US",
        }, ensure_ascii=False)

    @staticmethod
    def _normalize_aggregate_output(parsed: dict) -> dict:
        """Accept various JSON structures and normalize to {is_risk, risk_facts}."""
        # Already correct structure
        if "is_risk" in parsed and "risk_facts" in parsed:
            return parsed

        # Common variant: wrapped in fraud_risk_assessment
        wrappers = ["fraud_risk_assessment", "risk_assessment", "assessment",
                     "fraud_analysis", "analysis_result", "result"]
        inner = parsed
        for w in wrappers:
            if w in parsed and isinstance(parsed[w], dict):
                inner = parsed[w]
                break

        if "is_risk" in inner and "risk_facts" in inner:
            return inner

        # Extract risk_facts from key_findings / key_red_flags / red_flags / findings
        facts = (inner.get("risk_facts") or inner.get("key_findings") or
                 inner.get("key_red_flags") or inner.get("red_flags") or
                 inner.get("findings") or inner.get("fraud_indicators") or [])

        # Convert to standard risk_fact format
        normalized_facts = []
        for f in facts:
            if isinstance(f, dict):
                nf = {
                    "risk_title": str(f.get("risk_title") or f.get("indicator") or
                                      f.get("category") or f.get("title") or f.get("name", "")),
                    "involved_report": str(f.get("involved_report") or ""),
                    "confidence": float(f.get("confidence") or f.get("severity_score") or 0.5),
                    "evidence_chain": [],
                }
                # Extract evidence from details/observations/evidence
                details = (f.get("evidence_chain") or f.get("details") or
                          f.get("evidence") or f.get("observations") or [])
                if isinstance(details, str):
                    details = [{"point": details[:200], "analysis": details}]
                elif isinstance(details, list):
                    nf["evidence_chain"] = [
                        {"point": str(d)[:200], "analysis": str(d)}
                        for d in details[:5]
                    ]
                normalized_facts.append(nf)

        # Determine is_risk
        risk_level = str(inner.get("is_risk") or inner.get("risk_level") or
                        inner.get("overall_risk_level") or "").upper()
        is_risk = "1" if risk_level in ("HIGH", "MEDIUM", "1", "TRUE", "YES") else "0"
        if normalized_facts:
            is_risk = "1"

        return {"is_risk": is_risk, "risk_facts": normalized_facts[:5]}

    async def single_llm_aggregate(
        self,
        retrieved_text: str,
        fraud_period_str: str = "",
        session: aiohttp.ClientSession | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> str:
        """Ablation: single LLM call directly on retrieved text (no multi-expert).

        Used when multi_expert_enabled=False (no separate single/cross agents).
        """
        fp_info = fraud_period_str if fraud_period_str else "None specified"
        prompt = (
            f'Fill in the JSON template below based on the extracted report passages.\n\n'
            f'Known fraud period: {fp_info}\n\n'
            f'=== JSON TEMPLATE (fill this in) ===\n'
            f'{{"is_risk": "1", "risk_facts": ['
            f'{{"risk_title": "...", "involved_report": "...", "confidence": 0.0, '
            f'"evidence_chain": [{{"point": "...", "analysis": "..."}}]}}'
            f']}}\n\n'
            f'=== EXTRACTED REPORT PASSAGES ===\n{retrieved_text}\n=== END PASSAGES ===\n\n'
            f'Respond with the filled-in JSON template. '
            f'Valid JSON only, starting with {{"is_risk":.'
        )
        prompt = truncate_by_tokens(prompt, 60000)
        last_output = ""
        for attempt in range(self.max_json_retries):
            output = await self.chat(
                prompt, session, sem, temperature=0.1,
                track_phase="aggregation",
            )
            last_output = output[:300] if output else "(empty)"
            parsed = self.parse_json_response(output)
            if parsed:
                normalized = self._normalize_aggregate_output(parsed)
                if normalized.get("risk_facts") or normalized.get("is_risk") == "0":
                    normalized["market"] = "US"
                    return json.dumps(normalized, ensure_ascii=False, indent=2)
            logging.warning(
                f"Single LLM aggregate JSON parse attempt {attempt + 1}/{self.max_json_retries} "
                f"failed. Raw: {last_output}"
            )

        return json.dumps({
            "error": "Single LLM aggregate failed",
            "is_risk": "0",
            "risk_facts": [],
            "market": "US",
        }, ensure_ascii=False)
