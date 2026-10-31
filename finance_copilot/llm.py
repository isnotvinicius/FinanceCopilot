"""Criação do modelo de chat a partir da configuração.

`init_chat_model` é o único ponto que conhece o provedor. O agente recebe um `BaseChatModel`,
então trocar Anthropic por OpenAI é só uma mudança de variável de ambiente.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from finance_copilot.config import Settings


class LLMNotConfiguredError(RuntimeError):
    pass


def build_chat_model(settings: Settings) -> BaseChatModel:
    if not settings.llm_configured:
        raise LLMNotConfiguredError(
            f"Chave de API não encontrada para o provedor '{settings.llm_provider}'. "
            "Configure o arquivo .env (veja .env.example)."
        )
    kwargs = {}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # as novas tentativas são feitas pelo ModelRetryMiddleware
        **kwargs,
    )
