"""Particularidades de cada provedor, em um único lugar.

Todos os modelos são criados por `init_chat_model`; o que muda entre provedores são só os
argumentos. Nenhum outro módulo da aplicação testa o nome do provedor.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finance_copilot.config import PROVIDER_API_KEYS, Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    # Nome do provedor em init_chat_model (o OpenRouter usa o cliente da OpenAI).
    langchain_provider: str
    package: str
    api_key_env: str | None
    key_url: str
    # Argumentos próprios do provedor, calculados a partir da configuração.
    extra_kwargs: Callable[[Settings], dict[str, Any]] = field(default=lambda settings: {})
    # O cliente aceita timeout/max_retries como argumentos diretos.
    standard_client_options: bool = True

    @property
    def needs_api_key(self) -> bool:
        return self.api_key_env is not None

    def api_key(self, settings: Settings) -> str | None:
        attribute = PROVIDER_API_KEYS.get(self.id)
        return getattr(settings, attribute) if attribute else None


def _openrouter_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "base_url": OPENROUTER_BASE_URL,
        # Chat Completions: é a API que o OpenRouter implementa.
        "use_responses_api": False,
        "default_headers": {"X-Title": "Finance Copilot"},
        # Só roteia para endpoints que aceitam todos os parâmetros enviados (tools, tool_choice).
        # Sem isso, um provedor do pool pode ignorar as tools em silêncio.
        "extra_body": {"provider": {"require_parameters": True}},
    }


def _ollama_kwargs(settings: Settings) -> dict[str, Any]:
    # O ChatOllama não tem timeout próprio: ele vai para o cliente HTTP. Não há retries internos.
    return {
        "base_url": settings.ollama_base_url,
        "client_kwargs": {"timeout": settings.llm_timeout_seconds},
    }


PROVIDERS: dict[str, ProviderSpec] = {
    spec.id: spec
    for spec in [
        ProviderSpec(
            id="groq",
            label="Groq",
            langchain_provider="groq",
            package="langchain-groq",
            api_key_env="GROQ_API_KEY",
            key_url="https://console.groq.com/keys",
        ),
        ProviderSpec(
            id="google_genai",
            label="Google Gemini",
            langchain_provider="google_genai",
            package="langchain-google-genai",
            api_key_env="GOOGLE_API_KEY",
            key_url="https://aistudio.google.com/apikey",
        ),
        ProviderSpec(
            id="openrouter",
            label="OpenRouter",
            langchain_provider="openai",
            package="langchain-openai",
            api_key_env="OPENROUTER_API_KEY",
            key_url="https://openrouter.ai/settings/keys",
            extra_kwargs=_openrouter_kwargs,
        ),
        ProviderSpec(
            id="ollama",
            label="Ollama (local)",
            langchain_provider="ollama",
            package="langchain-ollama",
            api_key_env=None,
            key_url="https://ollama.com/download",
            extra_kwargs=_ollama_kwargs,
            standard_client_options=False,
        ),
        ProviderSpec(
            id="anthropic",
            label="Anthropic",
            langchain_provider="anthropic",
            package="langchain-anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            key_url="https://console.anthropic.com/settings/keys",
        ),
        ProviderSpec(
            id="openai",
            label="OpenAI",
            langchain_provider="openai",
            package="langchain-openai",
            api_key_env="OPENAI_API_KEY",
            key_url="https://platform.openai.com/api-keys",
        ),
    ]
}
