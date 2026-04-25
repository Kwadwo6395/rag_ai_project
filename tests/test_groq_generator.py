from unittest.mock import MagicMock, patch

from rag.generator import build_text_generator


@patch("rag.groq_generator.Groq")
def test_groq_generator_generate(mock_groq_cls):
    fake_client = MagicMock()
    mock_groq_cls.return_value = fake_client
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="  hello  "))]
    )
    from rag.groq_generator import GroqGenerator

    gen = GroqGenerator(api_key="test-key", model_name="llama-3.1-8b-instant")
    assert gen.generate("prompt") == "hello"
    fake_client.chat.completions.create.assert_called_once()
    call_kw = fake_client.chat.completions.create.call_args.kwargs
    assert call_kw["model"] == "llama-3.1-8b-instant"
    assert call_kw["messages"][0]["content"] == "prompt"


@patch("rag.groq_generator.GroqGenerator")
def test_build_text_factory_instantiates_groq(mock_groq_gen_cls):
    mock_groq_gen_cls.return_value = MagicMock()
    out = build_text_generator(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        groq_api_key="gk",
    )
    mock_groq_gen_cls.assert_called_once_with(
        api_key="gk",
        model_name="llama-3.1-8b-instant",
    )
    assert out is mock_groq_gen_cls.return_value


def test_build_text_factory_instantiates_gemini():
    with patch("rag.generator.GeminiGenerator") as mock_gem:
        mock_gem.return_value = MagicMock()
        build_text_generator(
            provider="gemini",
            model_id="gemini-2.5-flash-lite",
            gemini_api_key="gk",
        )
        mock_gem.assert_called_once_with(api_key="gk", model_name="gemini-2.5-flash-lite")
