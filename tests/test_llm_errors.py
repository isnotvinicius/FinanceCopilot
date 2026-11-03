"""Tradução das falhas de cada SDK de provedor em decisões de retry e mensagens ao usuário."""

import groq
import httpx
import ollama
import openai
import pytest
from google.genai.errors import ClientError
from langchain_google_genai.chat_models import GoogleRateLimitError

from finance_copilot.llm.errors import MAX_AUTOMATIC_WAIT_SECONDS, classify_llm_error
from tests.fakes import reply

THREAD = "t"


def http_error(cls, status: int, message: str, headers: dict | None = None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "http://local"))
    return cls(message, response=response, body=None)


def refused() -> Exception:
    """O que o ChatOllama lança com o servidor parado (verificado com o cliente real)."""
    try:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as cause:
            raise httpx.ConnectError("[Errno 111] Connection refused") from cause
    except httpx.ConnectError as exc:
        return exc


def gemini_error(message: str) -> Exception:
    """Como o langchain-google-genai entrega o erro: embrulhado, com a causa original encadeada."""
    cause = ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message}})
    try:
        raise GoogleRateLimitError(f"Error calling model (RESOURCE_EXHAUSTED): {cause}") from cause
    except GoogleRateLimitError as exc:
        return exc


@pytest.mark.parametrize(
    ("exc", "kind", "retryable"),
    [
        (http_error(groq.RateLimitError, 429, "Rate limit reached on requests per minute", {"retry-after": "3"}), "rate_limit", True),
        (http_error(groq.RateLimitError, 429, "Please try again in 1m30s (TPM)"), "rate_limit", False),
        (http_error(groq.RateLimitError, 429, "Rate limit reached on tokens per day (TPD)"), "quota", False),
        (http_error(groq.APIStatusError, 498, "Flex Tier Capacity Exceeded"), "server", True),
        (http_error(groq.AuthenticationError, 401, "Invalid API Key"), "auth", False),
        (http_error(openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-day"), "quota", False),
        (http_error(openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-min"), "rate_limit", True),
        (http_error(openai.APIStatusError, 402, "Insufficient credits"), "payment", False),
        (http_error(openai.NotFoundError, 404, "No endpoints found that support tool use"), "tool_unsupported", False),
        (http_error(openai.RateLimitError, 429, "You exceeded your current quota: insufficient_quota"), "quota", False),
        (gemini_error("Quota exceeded, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"), "quota", False),
        (gemini_error("Quota exceeded per minute. Please retry in 4.5s."), "rate_limit", True),
        (gemini_error("Quota exceeded per minute. Please retry in 45s."), "rate_limit", False),
        (ollama.ResponseError("registry.ollama.ai/library/gemma:2b does not support tools", 400), "tool_unsupported", False),
        (ollama.ResponseError("model 'qwen3:8b' not found, try pulling it first", 404), "not_found", False),
        (ConnectionError("Failed to connect to Ollama. Please check that Ollama is running"), "connection", False),
        (refused(), "connection", False),
        (ConnectionError("instável"), "unknown", True),
        (TimeoutError("timed out"), "timeout", True),
    ],
)
def test_classification(exc, kind, retryable):
    failure = classify_llm_error(exc)
    assert (failure.kind, failure.retryable) == (kind, retryable)


def test_wait_hint_is_parsed_and_capped():
    exc = http_error(groq.RateLimitError, 429, "Please try again in 1m2.5s")
    failure = classify_llm_error(exc)
    assert failure.retry_after == pytest.approx(62.5)
    assert failure.retry_after > MAX_AUTOMATIC_WAIT_SECONDS
    assert "cerca de 62 segundos" in failure.message


def test_auth_errors_keep_the_generic_configuration_message():
    assert classify_llm_error(http_error(groq.AuthenticationError, 401, "bad key")).message is None


def test_model_without_tools_is_reported_as_incompatible(make_session):
    error = ollama.ResponseError("registry.ollama.ai/library/gemma:2b does not support tools", 400)
    session = make_session(error, reply("nunca chega"), model_retries=2)

    result = session.send(THREAD, "Quanto gastei?")

    assert "não suporta tool calling" in result.error
    assert len(session.model.calls) == 1


def test_daily_quota_is_not_retried_and_thread_stays_usable(make_session):
    quota = http_error(groq.RateLimitError, 429, "Rate limit reached on tokens per day (TPD)")
    session = make_session(quota, reply("Voltei."), model_retries=2)

    result = session.send(THREAD, "oi")

    assert "cota do plano gratuito" in result.error
    assert "nunca troca de provedor" in result.error
    assert len(session.model.calls) == 1
    session.send(THREAD, "de novo")
    assert session.history(THREAD)[-1].text == "Voltei."


def test_ollama_offline_points_to_ollama_serve(make_session):
    session = make_session(ConnectionError("Failed to connect to Ollama."), model_retries=2)
    assert "ollama serve" in session.send(THREAD, "oi").error


def test_real_ollama_client_without_server_fails_fast(service, tmp_path):
    """Cliente real apontando para uma porta sem servidor: 1 tentativa e mensagem sobre o Ollama."""
    import socket

    from finance_copilot.agent.builder import build_agent, open_checkpointer
    from finance_copilot.agent.session import AgentSession
    from finance_copilot.config import Settings
    from finance_copilot.llm import build_chat_model

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    settings = Settings(
        _env_file=None, llm_provider="ollama", llm_model="qwen3:8b", ollama_base_url=f"http://127.0.0.1:{free_port}"
    )
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(build_chat_model(settings), saver, model_retries=2), service)

    assert "ollama serve" in session.send(THREAD, "oi").error
    saver.conn.close()


def groq_tool_use_failed(failed_generation: str, code: str = "tool_use_failed"):
    response = httpx.Response(
        400,
        json={"error": {"message": "Tool call validation failed", "code": code, "failed_generation": failed_generation}},
        request=httpx.Request("POST", "http://local"),
    )
    return groq.Groq(api_key="x")._make_status_error_from_response(response)


def test_rejected_tool_call_is_recovered_from_groq_error():
    from finance_copilot.llm.errors import recover_rejected_tool_calls

    exc = groq_tool_use_failed('{"name": "set_monthly_salary", "arguments": {"effective_from": "2026-11-01"}}')
    [recovered] = recover_rejected_tool_calls(exc)
    assert recovered["name"] == "set_monthly_salary"
    assert recovered["args"] == {"effective_from": "2026-11-01"}
    assert recovered["id"].startswith("call_")


@pytest.mark.parametrize(
    "exc",
    [
        groq_tool_use_failed("texto livre, não é JSON"),
        groq_tool_use_failed('{"name": "x"}'),
        groq_tool_use_failed('{"name": "x", "arguments": {}}', code="json_validate_failed"),
        http_error(groq.BadRequestError, 400, "context too long"),
    ],
)
def test_other_bad_requests_are_not_recovered(exc):
    from finance_copilot.llm.errors import recover_rejected_tool_calls

    assert recover_rejected_tool_calls(exc) is None
    failure = classify_llm_error(exc)
    assert (failure.kind, failure.retryable) == ("bad_request", False)
    assert ".env" not in failure.message
