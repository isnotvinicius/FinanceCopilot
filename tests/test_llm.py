"""Configuração da LLM, política de custo e formato real das requisições.

Os clientes reais (ChatAnthropic, ChatOpenAI, ChatGroq, ChatGoogleGenerativeAI, ChatOllama) falam
com um servidor HTTP local que imita a resposta dos provedores. Sem credenciais e sem rede, isso
confirma que as tools são serializadas e que o ciclo completo com aprovação funciona. Não avalia a
qualidade das decisões de um modelo real (veja tests/test_llm_live.py e scripts/smoke_llm.py).
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.agent.tools import ALL_TOOLS
from finance_copilot.bootstrap import build_app
from finance_copilot.config import Settings, configure_tracing
from finance_copilot.llm import (
    CostPolicyError,
    IncompatibleModelError,
    LLMNotConfiguredError,
    UnknownProviderError,
    build_chat_model,
    create_chat_model,
    selectable_models,
)
from finance_copilot.llm.catalog import ModelSpec, load_catalog
from finance_copilot.schemas import TransactionFilter

ARGS = {
    "kind": "expense",
    "amount": 85.5,
    "description": "Pizza",
    "category": "Alimentação",
    "occurred_on": "2026-10-14",
}
KEYS = ("anthropic_api_key", "openai_api_key", "groq_api_key", "google_api_key", "openrouter_api_key")


def make_settings(**overrides) -> Settings:
    defaults = {key: None for key in KEYS} | {"langsmith_api_key": None}
    return Settings(_env_file=None, **{**defaults, **overrides})


def write_catalog(tmp_path, entries: str):
    path = tmp_path / "models.toml"
    path.write_text(entries)
    return path


# --- Respostas no formato de cada provedor ----------------------------------------------------


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
    """Formato Chat Completions: usado por OpenAI, Groq e OpenRouter."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def gemini_response(part: dict) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }


def ollama_response(message: dict) -> dict:
    return {
        "model": "fake",
        "created_at": "2026-10-15T12:00:00Z",
        "message": {"role": "assistant", "content": "", **message},
        "done": True,
        "done_reason": "stop",
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
GEMINI_SCRIPT = [
    gemini_response({"functionCall": {"name": "create_transaction", "args": ARGS}}),
    gemini_response({"text": "Registrado!"}),
]
OLLAMA_SCRIPT = [
    ollama_response({"tool_calls": [{"function": {"name": "create_transaction", "arguments": ARGS}}]}),
    ollama_response({"content": "Registrado!"}),
]


@pytest.fixture
def provider_server():
    """Servidor local: responde com o roteiro em ordem e guarda os corpos recebidos.

    Um item do roteiro pode ser o corpo JSON (status 200) ou (status, headers, corpo).
    """
    requests: list[dict] = []
    script: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            requests.append(json.loads(self.rfile.read(size)))
            step = script.pop(0)
            status, headers, payload = step if isinstance(step, tuple) else (200, {}, step)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", script, requests
    server.shutdown()


def settings_for_server(tmp_path, url: str, provider: str, model: str, base_path: str = "") -> Settings:
    """Usa a fábrica real; só o endereço do provedor aponta para o servidor local.

    O endereço vai em `options` de um catálogo temporário, o mesmo mecanismo de ajuste por modelo
    usado em produção. O Ollama já recebe o endereço por OLLAMA_BASE_URL.
    """
    options = "" if provider == "ollama" else f'options = {{ base_url = "{url}{base_path}" }}'
    cost = "local" if provider == "ollama" else "free" if provider not in ("anthropic", "openai") else "paid"
    catalog = write_catalog(
        tmp_path,
        f"""
[[models]]
provider = "{provider}"
model = "{model}"
cost = "{cost}"
tool_calling = true
{options}
""",
    )
    return make_settings(
        llm_provider=provider,
        llm_model=model,
        llm_free_only=cost != "paid",
        llm_catalog_path=catalog,
        ollama_base_url=url,
        **{key: "test-key" for key in KEYS},
    )


FLOW_CASES = [
    ("anthropic", "fake-model", ANTHROPIC_SCRIPT, ""),
    ("openai", "fake-model", OPENAI_SCRIPT, "/v1"),
    ("groq", "openai/gpt-oss-120b", OPENAI_SCRIPT, ""),
    ("openrouter", "openrouter/free", OPENAI_SCRIPT, "/api/v1"),
    ("google_genai", "gemini-3.5-flash-lite", GEMINI_SCRIPT, ""),
    ("ollama", "qwen3:8b", OLLAMA_SCRIPT, ""),
]


@pytest.mark.parametrize(
    ("provider", "model_id", "script", "base_path"), FLOW_CASES, ids=[c[0] for c in FLOW_CASES]
)
def test_real_client_full_flow_with_approval(
    provider_server, service, tmp_path, provider, model_id, script, base_path
):
    url, queue, requests = provider_server
    queue.extend(script)
    model = build_chat_model(settings_for_server(tmp_path, url, provider, model_id, base_path))
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=0), service)

    assert session.send("t", "Gastei 85,50 com pizza ontem").error is None
    pending = session.pending_approval("t")
    assert pending.actions[0].name == "create_transaction"
    assert "R$ 85,50" in pending.actions[0].description
    assert service.list_transactions(TransactionFilter()).total_count == 0

    assert session.decide("t", pending.interrupt_id, approve=True).error is None

    assert service.list_transactions(TransactionFilter()).total_expenses.cents == 8550
    assert session.history("t")[-1].text == "Registrado!"

    sent = json.dumps(requests[0], ensure_ascii=False)
    assert all(tool.name in sent for tool in ALL_TOOLS)
    assert "runtime" not in sent
    # O resultado real da tool foi enviado ao provedor na segunda chamada.
    assert "R$ 85,50" in json.dumps(requests[1], ensure_ascii=False)
    saver.conn.close()


def test_rejection_flow_with_groq_client(provider_server, service, tmp_path):
    url, queue, requests = provider_server
    queue.extend([OPENAI_SCRIPT[0], openai_response({"role": "assistant", "content": "Ok, cancelado."}, "stop")])
    model = build_chat_model(settings_for_server(tmp_path, url, "groq", "openai/gpt-oss-120b"))
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=0), service)

    session.send("t", "Gastei 85,50 com pizza ontem")
    pending = session.pending_approval("t")
    session.decide("t", pending.interrupt_id, approve=False, reason="o valor foi 80")

    assert service.list_transactions(TransactionFilter()).total_count == 0
    assert session.history("t")[-1].tool_events[0].status == "rejected"
    assert "o valor foi 80" in json.dumps(requests[1], ensure_ascii=False)
    saver.conn.close()


@pytest.mark.parametrize(
    ("provider", "model_id", "status", "headers", "body", "expected"),
    [
        (
            "groq",
            "openai/gpt-oss-120b",
            429,
            {"retry-after": "30"},
            {"error": {"message": "Rate limit reached on tokens per minute (TPM)", "type": "tokens"}},
            "cerca de 30 segundos",
        ),
        (
            "groq",
            "openai/gpt-oss-120b",
            429,
            {},
            {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000"}},
            "cota do plano gratuito",
        ),
        (
            "openrouter",
            "openrouter/free",
            429,
            {},
            {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}},
            "cota do plano gratuito",
        ),
        (
            "openrouter",
            "openrouter/free",
            402,
            {},
            {"error": {"code": 402, "message": "Insufficient credits"}},
            "exige créditos",
        ),
        (
            "google_genai",
            "gemini-3.5-flash-lite",
            429,
            {},
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for metric generate_content_free_tier_requests, "
                    "limit: 250, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                }
            },
            "cota do plano gratuito",
        ),
    ],
    ids=["groq-long-wait", "groq-daily", "openrouter-daily", "openrouter-credits", "gemini-daily"],
)
def test_quota_and_rate_limit_errors_are_not_retried_and_explained(
    provider_server, service, tmp_path, provider, model_id, status, headers, body, expected
):
    """Um único pedido HTTP: nem o cliente nem o middleware repetem quando não vai adiantar."""
    url, queue, requests = provider_server
    queue.extend([(status, headers, body)] * 5)
    base_path = "/api/v1" if provider == "openrouter" else ""
    model = build_chat_model(settings_for_server(tmp_path, url, provider, model_id, base_path))
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=2), service)

    result = session.send("t", "Quanto gastei?")

    assert expected in result.error
    assert "Nenhuma operação foi realizada" in result.error or "nenhum modelo pago" in result.error
    assert len(requests) == 1
    saver.conn.close()


def test_short_rate_limit_is_retried_a_limited_number_of_times(provider_server, service, tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", waits.append)
    url, queue, requests = provider_server
    limited = (429, {"retry-after": "2"}, {"error": {"message": "Rate limit reached on requests per minute"}})
    queue.extend([limited] * 10)
    model = build_chat_model(settings_for_server(tmp_path, url, "groq", "openai/gpt-oss-120b"))
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=2), service)

    assert session.send("t", "oi").error is None

    assert len(requests) == 3  # 1 tentativa + 2 novas, nunca mais
    assert len(waits) == 2
    assert "limite de requisições" in session.history("t")[-1].text
    saver.conn.close()


# --- Fábrica: seleção, credenciais, custo e compatibilidade -----------------------------------


@pytest.mark.parametrize(
    ("provider", "model_id", "expected_class"),
    [
        ("groq", "openai/gpt-oss-120b", "ChatGroq"),
        ("google_genai", "gemini-3.5-flash-lite", "ChatGoogleGenerativeAI"),
        ("openrouter", "openrouter/free", "ChatOpenAI"),
        ("ollama", "qwen3:8b", "ChatOllama"),
    ],
)
def test_free_providers_are_created_from_catalog(provider, model_id, expected_class):
    settings = make_settings(llm_provider=provider, llm_model=model_id, **{key: "key" for key in KEYS})
    bundle = create_chat_model(settings)
    assert type(bundle.model).__name__ == expected_class
    assert bundle.spec.is_free
    assert bundle.spec.tool_calling is True


def test_client_options_disable_hidden_retries_and_set_timeout():
    settings = make_settings(llm_timeout_seconds=42, llm_temperature=0.1, **{key: "key" for key in KEYS})
    groq = build_chat_model(settings.with_llm("groq", "openai/gpt-oss-120b"))
    assert (groq.max_retries, groq.request_timeout, groq.temperature) == (0, 42, 0.1)
    gemini = build_chat_model(settings.with_llm("google_genai", "gemini-3.5-flash-lite"))
    assert (gemini.max_retries, gemini.timeout) == (0, 42)  # o padrão do cliente seria 6 retries
    ollama = build_chat_model(settings.with_llm("ollama", "qwen3:8b"))
    assert ollama.client_kwargs == {"timeout": 42}
    assert ollama.base_url == "http://localhost:11434"


def test_openrouter_uses_its_own_key_url_and_requires_tool_support():
    settings = make_settings(
        llm_provider="openrouter", llm_model="openrouter/free", openrouter_api_key="or-key", openai_api_key="oa-key"
    )
    model = build_chat_model(settings)
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "or-key"
    assert model.extra_body == {"provider": {"require_parameters": True}}
    assert "models" not in (model.extra_body or {})  # sem fallback para outros (pagos) modelos


def test_openrouter_never_reuses_openai_key():
    settings = make_settings(llm_provider="openrouter", llm_model="openrouter/free", openai_api_key="oa-key")
    with pytest.raises(LLMNotConfiguredError, match="OPENROUTER_API_KEY"):
        build_chat_model(settings)


@pytest.mark.parametrize(
    ("provider", "model_id", "env_name"),
    [
        ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY"),
        ("google_genai", "gemini-3.5-flash-lite", "GOOGLE_API_KEY"),
        ("openrouter", "openrouter/free", "OPENROUTER_API_KEY"),
    ],
)
def test_missing_credentials_name_the_variable(provider, model_id, env_name):
    with pytest.raises(LLMNotConfiguredError, match=env_name):
        build_chat_model(make_settings(llm_provider=provider, llm_model=model_id))


def test_ollama_needs_no_api_key():
    settings = make_settings(llm_provider="ollama", llm_model="qwen3:8b")
    assert settings.llm_configured
    assert type(build_chat_model(settings)).__name__ == "ChatOllama"


def test_build_chat_model_requires_api_key():
    with pytest.raises(LLMNotConfiguredError, match="anthropic"):
        build_chat_model(make_settings(llm_provider="anthropic", llm_free_only=False))


def test_blank_env_values_count_as_missing():
    settings = make_settings(anthropic_api_key="", llm_temperature="", llm_catalog_path="")
    assert settings.llm_provider == "groq"
    assert settings.llm_configured is False
    assert settings.llm_temperature is None
    assert settings.llm_catalog_path is None


@pytest.mark.parametrize(
    ("provider", "key_field", "expected_class"),
    [("anthropic", "anthropic_api_key", "ChatAnthropic"), ("openai", "openai_api_key", "ChatOpenAI")],
)
def test_provider_is_selected_by_configuration(provider, key_field, expected_class):
    settings = make_settings(
        llm_provider=provider, llm_model="any-model", llm_free_only=False, **{key_field: "key"}
    )
    assert type(build_chat_model(settings)).__name__ == expected_class


def test_selection_comes_from_env_file(tmp_path, monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_FREE_ONLY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "LLM_PROVIDER=google_genai\nLLM_MODEL=gemini-3.6-flash\nLLM_FREE_ONLY=true\n"
        "GOOGLE_API_KEY=g-key\nLLM_TIMEOUT_SECONDS=15\n"
    )
    settings = Settings(_env_file=env)
    model = build_chat_model(settings)
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model.endswith("gemini-3.6-flash")
    assert model.timeout == 15


def test_unknown_provider_is_rejected_with_the_valid_options():
    with pytest.raises(UnknownProviderError, match="groq, google_genai, openrouter, ollama"):
        build_chat_model(make_settings(llm_provider="azure_free", llm_model="x"))


@pytest.mark.parametrize(
    ("provider", "model_id", "reason"),
    [
        ("anthropic", "claude-sonnet-5-5", "é um modelo pago"),
        ("groq", "llama-3.3-70b-versatile", "não está no catálogo"),
        ("openrouter", "openai/gpt-5", "não está no catálogo"),
        ("openrouter", "some/model:free", "não está no catálogo"),
        ("ollama", "gpt-oss:120b-cloud", "não está no catálogo"),
    ],
)
def test_free_mode_blocks_paid_and_unverified_models(provider, model_id, reason):
    settings = make_settings(llm_provider=provider, llm_model=model_id, **{key: "key" for key in KEYS})
    with pytest.raises(CostPolicyError, match=reason):
        build_chat_model(settings)


def test_blocked_model_never_falls_back_to_another(tmp_path):
    settings = make_settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-5-5",
        finance_db_path=tmp_path / "f.db",
        agent_state_db_path=tmp_path / "s.db",
        **{key: "key" for key in KEYS},
    )
    app = build_app(settings)
    assert app.model is None and app.session is None
    assert "LLM_FREE_ONLY=true" in app.llm_error


def test_free_mode_ui_hides_paid_models():
    free = selectable_models(make_settings())
    assert free and all(spec.is_free for spec in free)
    everything = selectable_models(make_settings(llm_free_only=False))
    assert any(spec.cost == "paid" for spec in everything)


def test_catalog_rejects_model_without_tool_calling(tmp_path):
    catalog = write_catalog(
        tmp_path,
        '[[models]]\nprovider = "groq"\nmodel = "chat-only"\ncost = "free"\ntool_calling = false\n',
    )
    settings = make_settings(llm_model="chat-only", llm_catalog_path=catalog, groq_api_key="k")
    with pytest.raises(IncompatibleModelError, match="tool calling"):
        build_chat_model(settings)


def test_profile_without_tool_calling_is_rejected(monkeypatch):
    from tests.fakes import scripted

    fake = scripted()
    fake.profile = {"tool_calling": False}
    monkeypatch.setattr("finance_copilot.llm.factory.init_chat_model", lambda *a, **k: fake)
    settings = make_settings(llm_free_only=False, llm_model="unlisted-model", groq_api_key="k")
    with pytest.raises(IncompatibleModelError, match="unlisted-model"):
        build_chat_model(settings)


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"provider": "openrouter", "model": "openai/gpt-5", "cost": "free"}, "só openrouter/free"),
        ({"provider": "anthropic", "model": "claude", "cost": "free"}, "não tem camada gratuita"),
        ({"provider": "ollama", "model": "gpt-oss:120b-cloud", "cost": "local"}, "localmente"),
        ({"provider": "groq", "model": "x", "cost": "local"}, "localmente"),
    ],
)
def test_catalog_refuses_wrong_cost_labels(entry, error):
    with pytest.raises(ValueError, match=error):
        ModelSpec.model_validate(entry)


def test_shipped_catalog_is_consistent():
    catalog = load_catalog(make_settings())
    assert len({spec.key for spec in catalog}) == len(catalog)
    for spec in catalog:
        assert spec.tool_calling is True, spec.key
        if spec.is_free:
            assert spec.checked_on is not None, spec.key
    recommended = [spec for spec in catalog if spec.recommended and spec.is_free]
    assert {spec.provider for spec in recommended} == {"groq", "google_genai", "openrouter", "ollama"}
    # O padrão do .env.example precisa ser um modelo gratuito do catálogo.
    default = make_settings()
    assert any(s.key == f"{default.llm_provider}/{default.llm_model}" and s.is_free for s in catalog)


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


DRAFT = {"kind": "expense", "amount": 32.9, "description": "Uber", "category": "Transporte", "occurred_on": "2026-10-15"}
DRAFT_TOOL_CALL = openai_response(
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "TransactionDraft", "arguments": json.dumps(DRAFT)}}
        ],
    },
    "tool_calls",
)
EXTRACTION_CASES = [
    ("groq", "openai/gpt-oss-120b", DRAFT_TOOL_CALL, "", "tools"),
    ("openrouter", "openrouter/free", DRAFT_TOOL_CALL, "/api/v1", "tools"),
    ("google_genai", "gemini-3.5-flash-lite", gemini_response({"text": json.dumps(DRAFT)}), "", "generationConfig"),
    ("ollama", "qwen3:8b", ollama_response({"content": json.dumps(DRAFT)}), "", "format"),
]


@pytest.mark.parametrize(
    ("provider", "model_id", "response", "base_path", "request_field"),
    EXTRACTION_CASES,
    ids=[c[0] for c in EXTRACTION_CASES],
)
def test_structured_output_uses_catalog_method_with_real_client(
    provider_server, service, tmp_path, provider, model_id, response, base_path, request_field
):
    from finance_copilot.agent.extraction import extract_transaction_draft
    from finance_copilot.llm.catalog import find_model

    url, queue, requests = provider_server
    queue.append(response)
    settings = settings_for_server(tmp_path, url, provider, model_id, base_path)
    method = find_model(make_settings(), provider, model_id).structured_output_method

    draft = extract_transaction_draft(build_chat_model(settings), service, "Uber de 32,90 hoje", method=method)

    assert (str(draft.amount), draft.category) == ("32.9", "Transporte")
    # O método do catálogo chegou ao provedor: tool forçada, schema JSON nativo ou "format".
    assert request_field in requests[0]


def test_groq_server_side_tool_validation_error_lets_model_correct_itself(provider_server, service, tmp_path):
    """O Groq valida as tool calls no servidor e recusa a resposta inteira (400 tool_use_failed).
    A chamada recusada volta ao fluxo normal: a validação local devolve o erro ao modelo, sem
    gravar nada, e ele corrige o argumento. Caso real: effective_from '2026-11-01' em vez de '2026-11'."""
    url, queue, requests = provider_server
    rejected = {
        "error": {
            "message": "Tool call validation failed: parameters for tool set_monthly_salary did not match schema",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": json.dumps(
                {"name": "set_monthly_salary", "arguments": {"amount": 9261.0, "effective_from": "2026-11-01"}}
            ),
        }
    }
    corrected = openai_response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "set_monthly_salary",
                        "arguments": json.dumps({"amount": 9261.0, "effective_from": "2026-11"}),
                    },
                }
            ],
        },
        "tool_calls",
    )
    queue.extend([(400, {}, rejected), corrected, openai_response({"role": "assistant", "content": "Salário salvo."}, "stop")])
    model = build_chat_model(settings_for_server(tmp_path, url, "groq", "openai/gpt-oss-120b"))
    saver = open_checkpointer(tmp_path / "state.db")
    session = AgentSession(build_agent(model, saver, model_retries=2), service)

    assert session.send("t", "Meu salário agora é 9261 a partir de novembro").error is None
    # O argumento inválido não gerou cartão de aprovação nem gravação; o erro foi para o modelo.
    assert "effective_from" in json.dumps(requests[1], ensure_ascii=False)
    pending = session.pending_approval("t")
    assert pending.actions[0].args["effective_from"] == "2026-11"
    assert service.get_salary("2026-11") is None

    session.decide("t", pending.interrupt_id, approve=True)

    assert service.get_salary("2026-11").cents == 926100
    assert session.history("t")[-1].text == "Salário salvo."
    assert len(requests) == 3
    saver.conn.close()
