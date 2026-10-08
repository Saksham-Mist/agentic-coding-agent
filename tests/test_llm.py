"""Step 3.1 done-when check: tokens per call are logged.

Makes a real (small, cheap) call to the Gemini free tier, so it skips
when GEMINI_API_KEY (local-only, in .env) isn't available rather than
failing a fresh clone's test run.
"""
import logging
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from agentic_coding_agent.agent.llm import GeminiClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


def test_generate_logs_token_usage(caplog):
    client = GeminiClient()
    with caplog.at_level(logging.INFO, logger="agentic_coding_agent.llm"):
        result = client.generate("Reply with exactly one word: OK")

    assert result["text"]
    assert result["usage"]["total_tokens"] > 0

    log_messages = [r.message for r in caplog.records]
    assert any("gemini call:" in m and "total_tokens=" in m for m in log_messages)
