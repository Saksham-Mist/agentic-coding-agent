"""Gemini client wrapper — step 3.1.

Every call logs token usage so the design notes' 100k-tokens/run budget
(step 3.5) has real numbers to check against, not estimates.
"""
from __future__ import annotations

import logging
import os

from google import genai
from google.genai.errors import ServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("agentic_coding_agent.llm")

# Configurable because free-tier quota is per model (20 requests/day
# on gemini-3.8-flash vs 500 on 3.5-flash-lite); override via GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

# The free tier returns 503 "high demand" often enough in practice
# (hit twice in three calls while building this) that retrying is
# load-bearing, not defensive polish — found while verifying this
# step, see the design notes section 7.
_RETRY = retry(
    retry=retry_if_exception_type(ServerError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)


class GeminiClient:
    """Thin wrapper around google-genai. One `generate` call = one
    logged call, with prompt/response/thoughts/total token counts."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str, system_instruction: str | None = None) -> dict:
        """Returns {text, usage: {prompt_tokens, response_tokens,
        thoughts_tokens, total_tokens}}. Retries on transient 503s
        (free-tier "high demand") with exponential backoff; raises on
        anything else — the caller (Planner/Worker nodes, step
        3.2/3.3) is responsible for turning that into a structured
        retry/abort decision, same rule as every tool in
        the design notes."""
        response = self._generate_with_retry(prompt, system_instruction)

        usage = response.usage_metadata
        token_usage = {
            "prompt_tokens": usage.prompt_token_count or 0,
            "response_tokens": usage.candidates_token_count or 0,
            "thoughts_tokens": usage.thoughts_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }
        logger.info(
            "gemini call: model=%s prompt_tokens=%d response_tokens=%d "
            "thoughts_tokens=%d total_tokens=%d",
            self._model,
            token_usage["prompt_tokens"],
            token_usage["response_tokens"],
            token_usage["thoughts_tokens"],
            token_usage["total_tokens"],
        )

        return {"text": response.text, "usage": token_usage}

    @_RETRY
    def _generate_with_retry(self, prompt: str, system_instruction: str | None):
        config = {"system_instruction": system_instruction} if system_instruction else None
        return self._client.models.generate_content(
            model=self._model, contents=prompt, config=config,
        )
