"""Base agent with shared LLM invocation, retry, and token truncation logic."""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
import tiktoken

ENCODING_NAME = "cl100k_base"
# TODO: Set your API keys via environment variables before running.
# Original keys backed up in api_keys_backup.txt (EXCLUDED from git).
API_CONFIGS = {
    "minimax-m2.5": {
        "url": "https://api.minimax.chat/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "deepseek-v4": {
        "url": "https://api.deepseek.com/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "deepseek-reasoner": {
        "url": "https://api.deepseek.com/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "gpt-4o": {
        "url": "https://api.vectorengine.ai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
    "o3-mini": {
        "url": "https://api.vectorengine.ai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    },
}
MODEL_CONFIGS = {
    "minimax-m2.5": {"model": "MiniMax-M2.5"},
    "deepseek-v4": {"model": "deepseek-v4-pro"},
    "deepseek-reasoner": {"model": "deepseek-reasoner"},
    "gpt-4o": {"model": "gpt-4o"},
    "o3-mini": {"model": "o3-mini"},
}
DEFAULT_SYSTEM_ROLE = "You are a financial auditing expert."


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


class BaseAgent:
    """Shared base for all pipeline agents."""

    def __init__(
        self,
        model_name: str = "minimax-m2.5",
        system_role: str = DEFAULT_SYSTEM_ROLE,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.system_role = system_role
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.token_usage: dict[str, int] = {}  # {phase: total_prompt_tokens}

    def _count_tokens(self, text: str) -> int:
        return count_tokens(text)

    async def chat(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore | None = None,
        temperature: float = 0.6,
        track_phase: str = "",
        response_format: dict | None = None,
    ) -> str:
        """Call LLM with exponential backoff retry.

        Args:
            track_phase: If set, accumulates actual API token usage into token_usage[track_phase].
            response_format: Optional OpenAI-style response_format dict (e.g. {"type": "json_object"}).
        """
        if sem:
            async with sem:
                return await self._call(prompt, session, temperature, track_phase, response_format)
        return await self._call(prompt, session, temperature, track_phase, response_format)

    async def _call(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
        temperature: float = 0.6,
        track_phase: str = "",
        response_format: dict | None = None,
    ) -> str:
        api_cfg = API_CONFIGS.get(self.model_name, API_CONFIGS["minimax-m2.5"])
        model_cfg = MODEL_CONFIGS.get(self.model_name, MODEL_CONFIGS["minimax-m2.5"])
        truncated = truncate_by_tokens(prompt, 60000)

        for attempt in range(self.max_retries + 1):
            try:
                payload = {
                    "model": model_cfg["model"],
                    "messages": [
                        {"role": "system", "content": self.system_role},
                        {"role": "user", "content": truncated},
                    ],
                    "temperature": temperature,
                    "top_p": 0.9,
                }
                # skip_think is MiniMax-specific; skip for other providers
                if self.model_name.startswith("minimax"):
                    payload["skip_think"] = True
                if response_format:
                    payload["response_format"] = response_format

                async with session.post(
                    api_cfg["url"],
                    headers=api_cfg["headers"],
                    json=payload,
                ) as resp:
                    if resp.status == 429:
                        wait = min(300, (2 ** attempt) * 5)
                        logging.warning(f"429 — waiting {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue
                    data = await resp.json()
                    if "error" in data:
                        logging.error(f"API error: {data['error']}")
                        if attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return ""

                    # Track actual API token usage
                    if track_phase:
                        usage = data.get("usage", {})
                        actual_tokens = usage.get("total_tokens", 0)
                        if actual_tokens:
                            self.token_usage[track_phase] = (
                                self.token_usage.get(track_phase, 0) + actual_tokens
                            )

                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content:
                        logging.warning(f"LLM returned empty content (attempt {attempt + 1})")
                        return content

                    # Strip thinking tags (MiniMax M2.5 may wrap reasoning in <think>...</think>)
                    think_end = content.find("</think>")
                    if think_end != -1:
                        content = content[think_end + 8:].strip()
                    return content
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.error(f"HTTP error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(60, 2 ** attempt))
                else:
                    return ""
        return ""

    @staticmethod
    def _repair_json(text: str) -> str:
        """Fix common LLM JSON errors: trailing commas, unquoted values."""
        import re
        # Remove trailing comma before closing brace/bracket
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text

    def parse_json_response(self, content: str) -> dict:
        """Extract first balanced JSON object from LLM output."""
        # Strip thinking tags (safety net — normally done in _call)
        think_end = content.find("</think>")
        if think_end != -1:
            content = content[think_end + 8:].strip()
        cleaned = content.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        if start == -1:
            logging.warning(f"parse_json_response: no JSON object found in: {content[:200]}")
            return {}
        depth = 0
        for i, ch in enumerate(cleaned[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cleaned = cleaned[start:i + 1]
                    break
        else:
            cleaned = cleaned[start:]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = self._repair_json(cleaned)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                logging.warning(
                    f"parse_json_response failed after repair. "
                    f"Raw (first 300): {content[:300]}"
                )
                return {}
