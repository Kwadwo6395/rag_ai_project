# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/groq_generator.py
"""Groq OpenAI-compatible chat backend (same .generate() contract as GeminiGenerator)."""
from __future__ import annotations

import os
import re
import time

try:
    from groq import Groq
except ImportError as e:  # pragma: no cover
    Groq = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

_RETRY_AFTER_RE = re.compile(r"retry after (\d+)", re.I)


class GroqGenerator:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        if Groq is None:
            raise RuntimeError(
                "The `groq` package is not installed. Run: pip install groq"
            ) from _IMPORT_ERROR
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        default_model = "llama-3.1-8b-instant"
        resolved = (model_name or os.environ.get("GROQ_MODEL") or default_model).strip()
        self.model_id = resolved or default_model
        self._client = Groq(api_key=api_key)
        self._max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "8192"))

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        max_attempts = max(1, int(os.environ.get("GROQ_MAX_RETRIES", "3")))
        for attempt in range(max_attempts):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self._max_tokens,
                )
                msg = completion.choices[0].message.content
                return (msg or "").strip()
            except Exception as e:
                err = str(e).lower()
                is_rate = "429" in str(e) or "rate" in err or "quota" in err
                if not is_rate or attempt + 1 >= max_attempts:
                    raise RuntimeError(
                        f"Groq API error ({self.model_id}): {e}\n"
                        "See https://console.groq.com/docs/errors"
                    ) from e
                m = _RETRY_AFTER_RE.search(str(e))
                wait = float(m.group(1)) if m else min(5.0 * (2**attempt), 45.0)
                time.sleep(wait + 0.5)
