"""Criação do modelo de chat: provedores, catálogo de modelos, política de custo e erros."""

from finance_copilot.llm.catalog import ModelSpec, resolve_model, selectable_models
from finance_copilot.llm.errors import (
    CostPolicyError,
    IncompatibleModelError,
    LLMConfigurationError,
    LLMNotConfiguredError,
    UnknownProviderError,
    classify_llm_error,
)
from finance_copilot.llm.factory import ChatModelBundle, build_chat_model, create_chat_model

__all__ = [
    "ChatModelBundle",
    "CostPolicyError",
    "IncompatibleModelError",
    "LLMConfigurationError",
    "LLMNotConfiguredError",
    "ModelSpec",
    "UnknownProviderError",
    "build_chat_model",
    "classify_llm_error",
    "create_chat_model",
    "resolve_model",
    "selectable_models",
]
