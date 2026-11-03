"""Fábrica do modelo de chat: configuração -> validação -> `init_chat_model`.

É o único ponto que cria modelos. O agente recebe um `BaseChatModel`, então trocar de provedor
é só uma mudança de variável de ambiente (ou da seleção na barra lateral).
"""

from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from finance_copilot.config import Settings
from finance_copilot.llm.catalog import ModelSpec, resolve_model
from finance_copilot.llm.errors import (
    IncompatibleModelError,
    LLMNotConfiguredError,
    ProviderPackageMissingError,
)
from finance_copilot.llm.providers import PROVIDERS


@dataclass(frozen=True)
class ChatModelBundle:
    model: BaseChatModel
    spec: ModelSpec


def create_chat_model(settings: Settings) -> ChatModelBundle:
    """Cria o modelo configurado ou lança LLMConfigurationError com uma mensagem clara.

    Nenhuma requisição é feita aqui: a validação usa só a configuração e o catálogo.
    """
    spec = resolve_model(settings)
    provider = PROVIDERS[spec.provider]
    kwargs = {}
    if provider.needs_api_key:
        api_key = provider.api_key(settings)
        if not api_key:
            raise LLMNotConfiguredError(
                f"Chave de API não encontrada para o provedor '{spec.provider}'. Defina "
                f"{provider.api_key_env} no arquivo .env (veja .env.example) e reinicie. "
                f"Crie a chave em {provider.key_url}."
            )
        kwargs["api_key"] = api_key
    if provider.standard_client_options:
        kwargs["timeout"] = settings.llm_timeout_seconds
        # As novas tentativas são feitas só pelo ModelRetryMiddleware, com limite e backoff.
        # Os clientes têm retries próprios (o do Gemini faz 6), que gastariam cota em silêncio.
        kwargs["max_retries"] = 0
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    kwargs |= provider.extra_kwargs(settings) | spec.options

    try:
        model = init_chat_model(spec.model, model_provider=provider.langchain_provider, **kwargs)
    except ImportError as exc:
        raise ProviderPackageMissingError(
            f"Pacote da integração '{spec.provider}' não instalado. Rode: "
            f".venv/bin/python -m pip install -r requirements.txt ({provider.package})."
        ) from exc

    if spec.tool_calling is None and (getattr(model, "profile", None) or {}).get("tool_calling") is False:
        raise IncompatibleModelError(
            f"O perfil do LangChain indica que '{spec.model}' não suporta tool calling, "
            "recurso essencial do agente. Escolha outro modelo."
        )
    return ChatModelBundle(model, spec)


def build_chat_model(settings: Settings) -> BaseChatModel:
    return create_chat_model(settings).model
