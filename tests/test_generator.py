from unittest.mock import patch, MagicMock
from google.api_core import exceptions as google_api_exceptions
from rag.generator import GeminiGenerator


def test_generate_calls_model_with_prompt():
    fake_model = MagicMock()
    fake_model.generate_content.return_value = MagicMock(text="answer")
    with patch("rag.generator.genai.GenerativeModel", return_value=fake_model):
        gen = GeminiGenerator(api_key="fake-key")
        out = gen.generate("hello")
    assert out == "answer"
    fake_model.generate_content.assert_called_once()
    args, _ = fake_model.generate_content.call_args
    assert args[0] == "hello"


def test_generate_retries_resource_exhausted_then_ok():
    exhausted = google_api_exceptions.ResourceExhausted(
        "429 Please retry in 0.01s."
    )
    fake_model = MagicMock()
    fake_model.generate_content.side_effect = [
        exhausted,
        MagicMock(text="ok"),
    ]
    with patch("rag.generator.genai.GenerativeModel", return_value=fake_model), \
            patch("rag.generator.time.sleep"):
        gen = GeminiGenerator(api_key="fake-key")
        out = gen.generate("hello")
    assert out == "ok"
    assert fake_model.generate_content.call_count == 2
