# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/generator.py
from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions
from google.generativeai.types import RequestOptions

if TYPE_CHECKING:
    from rag.groq_generator import GroqGenerator

_RETRY_AFTER_RE = re.compile(r"retry in ([0-9.]+)\s*s", re.I)
# Default: Flash-Lite has its own free-tier quota bucket vs gemini-2.5-flash (fewer 429s in practice).
_DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _safe_response_text(resp) -> str:
    """Avoid hard failures when the API returns no text (safety block, etc.)."""
    try:
        return (resp.text or "").strip()
    except ValueError as e:
        return (
            "The model did not return usable text (it may have been blocked or filtered). "
            f"Try rephrasing your question. Technical detail: {e}"
        )


class GeminiGenerator:
    def __init__(self, api_key: str | None = None,
                 model_name: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        resolved = (model_name or os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL)
        self.model_id = str(resolved).strip() or _DEFAULT_MODEL
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_id)
        self._request_options = RequestOptions(
            timeout=float(os.environ.get("GEMINI_TIMEOUT_SEC", "120")),
        )

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Fewer retries + capped sleep so the UI does not look "stuck" for many minutes on 429.
        max_attempts = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "3")))
        for attempt in range(max_attempts):
            try:
                resp = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                    request_options=self._request_options,
                )
                return _safe_response_text(resp)
            except google_api_exceptions.ResourceExhausted as e:
                if attempt + 1 >= max_attempts:
                    raise RuntimeError(
                        f"{e}\n\nWhat this means: on the **free** Gemini API, Google caps how many "
                        "`generate_content` calls you can make per day **per model** (often ~20). "
                        "This app uses **one** call per question by default (answer only); "
                        "enable 'LLM query planning' in the sidebar for two calls when you need it.\n\n"
                        f"**This process is using model:** `{self.model_id}`.\n"
                        "If you expected `gemini-2.5-flash-lite` but errors mention `gemini-2.5-flash`, "
                        "check `echo $GEMINI_MODEL` in the terminal — we now load `.env` with override so "
                        "`.env` should win; restart Streamlit after changing `.env`.\n\n"
                        "Fixes: wait for the daily reset, create a **new API key/project**, set "
                        "`GEMINI_MODEL` to another supported model, or enable **billing**. "
                        "https://ai.google.dev/gemini-api/docs/rate-limits"
                    ) from e
                msg = str(e)
                m = _RETRY_AFTER_RE.search(msg)
                if m:
                    wait = float(m.group(1))
                else:
                    wait = min(20.0 * (1.2**attempt), 60.0)
                time.sleep(min(wait + 1.0, 50.0))


def build_text_generator(
    *,
    provider: str,
    model_id: str | None = None,
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
) -> GeminiGenerator | GroqGenerator:
    """Factory: ``provider`` is ``gemini`` or ``groq`` (case-insensitive)."""
    from rag.groq_generator import GroqGenerator

    p = (provider or "gemini").strip().lower()
    if p == "groq":
        mid = (model_id or os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        return GroqGenerator(api_key=groq_api_key, model_name=mid)
    mid = (model_id or os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL).strip()
    return GeminiGenerator(api_key=gemini_api_key, model_name=mid or None)
