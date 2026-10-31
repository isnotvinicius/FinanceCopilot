"""Configuração da LLM e formato real das requisições.

Os clientes reais (ChatAnthropic, ChatOpenAI) falam com um servidor HTTP local que imita a resposta
dos provedores. Sem credenciais e sem rede, isso confirma que as tools são serializadas e que o
ciclo completo com aprovação funciona. Não avalia a qualidade das decisões de um modelo real.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain.chat_models import init_chat_model

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.config import Settings, configure_tracing
from finance_copilot.llm import LLMNotConfiguredError, build_chat_model
from finance_copilot.schemas import TransactionFilter

ARGS = {
    "kind": "expense",
    "amount": 85.5,
    "description": "Pizza",
    "category": "Alimentação",
    "occurred_on": "2026-10-14",
}


def make_settings(**overrides) -> Settings:
    defaults = {"anthropic_api_key": None, "openai_api_key": None, "langsmith_api_key": None}
    return Settings(_env_file=None, **{**defaults, **overrides})


def anthropic_response(block: dict, stop_reason: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "fake",
        "content": [block],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def openai_response(message: dict, finish_reason: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


ANTHROPIC_SCRIPT = [
    anthropic_response(
        {"type": "tool_use", "id": "toolu_1", "name": "create_transaction", "input": ARGS}, "tool_use"
    ),
    anthropic_response({"type": "text", "text": "Registrado!"}, "end_turn"),
]
OPENAI_SCRIPT = [
    openai_response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "create_transaction", "arguments": json.dumps(ARGS)},
                }
            ],
        },
        "tool_calls",
    ),
    openai_response({"role": "assistant", "content": "Registrado!"}, "stop"),
]


@pytest.fixture
def provider_server():
    """Servidor local: responde com o roteiro em ordem e guarda os corpos recebidos."""
    requests: list[dict] = []
    script: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            requests.append(json.loads(self.rfile.read(size)))
            body = json.dumps(script.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", script, requests
    server.shutdown()


@pytest.mark.parametrize(
    ("provider", "script", "base_path"),
    [("anthropic", ANTHROPIC_SCRIPT, ""), ("openai", OPENAI_SCRIPT, "/v1")],
)
def test_real_client_full_flow_with_approval(provider_server, service, tmp_path, provider, script, base_path):
    url, queue, requests = provider_server
    queue.extend(script)
    model = init_chat_model(
        "fake-model",
        model_provider=provider,
        api_key="test-key",
        base_url=url + base_path,
        max_retries=0,
    )
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=0), service)

    assert session.send("t", "Gastei 85,50 com pizza ontem").error is None
    pending = session.pending_approval("t")
    assert "R$ 85,50" in pending.actions[0].description
    assert service.list_transactions(TransactionFilter()).total_count == 0

    assert session.decide("t", pending.interrupt_id, approve=True).error is None

    assert service.list_transactions(TransactionFilter()).total_expenses.cents == 8550
    assert session.history("t")[-1].text == "Registrado!"

    sent_tools = json.dumps(requests[0]["tools"])
    assert "create_transaction" in sent_tools
    assert "simulate_budget" in sent_tools
    assert "runtime" not in sent_tools
    assert len(requests[0]["tools"]) == 9
    # O resultado real da tool foi enviado ao provedor na segunda chamada.
    assert "R$ 85,50" in json.dumps(requests[1]["messages"], ensure_ascii=False)
    saver.conn.close()


def test_build_chat_model_requires_api_key():
    with pytest.raises(LLMNotConfiguredError, match="anthropic"):
        build_chat_model(make_settings(llm_provider="anthropic"))


def test_blank_env_values_count_as_missing():
    settings = make_settings(anthropic_api_key="", llm_temperature="")
    assert settings.llm_configured is False
    assert settings.llm_temperature is None


@pytest.mark.parametrize(
    ("provider", "key_field", "expected_class"),
    [("anthropic", "anthropic_api_key", "ChatAnthropic"), ("openai", "openai_api_key", "ChatOpenAI")],
)
def test_provider_is_selected_by_configuration(provider, key_field, expected_class):
    settings = make_settings(llm_provider=provider, llm_model="any-model", **{key_field: "key"})
    assert type(build_chat_model(settings)).__name__ == expected_class


def test_tracing_requires_flag_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert configure_tracing(make_settings(langsmith_tracing=False)) is False
    assert configure_tracing(make_settings(langsmith_tracing=True)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert configure_tracing(make_settings(langsmith_tracing=True, langsmith_api_key="k")) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
