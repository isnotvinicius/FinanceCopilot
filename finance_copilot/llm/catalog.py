"""Catálogo de modelos e política de custo (LLM_FREE_ONLY).

O catálogo (models.toml) diz o que se sabe de cada modelo: custo, tool calling e o método de
structured output. A política é deliberadamente conservadora: no modo gratuito, só passa o que
está no catálogo como "free" ou "local". Custo desconhecido é tratado como pago.
"""

import hashlib
import tomllib
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from finance_copilot.config import Settings
from finance_copilot.llm.errors import CostPolicyError, IncompatibleModelError, UnknownProviderError
from finance_copilot.llm.providers import PROVIDERS

DEFAULT_CATALOG_PATH = Path(__file__).with_name("models.toml")

Cost = Literal["free", "local", "paid", "unknown"]
StructuredOutput = Literal["function_calling", "json_schema", "json_mode", "default", "none"]

COST_LABELS = {
    "free": "Gratuito (camada free)",
    "local": "Local",
    "paid": "Pago",
    "unknown": "Custo não verificado",
}
FREE_COSTS = ("free", "local")
PAID_ONLY_PROVIDERS = {"anthropic", "openai"}


class ModelSpec(BaseModel):
    provider: str
    model: str
    label: str = ""
    cost: Cost = "unknown"
    recommended: bool = False
    # None: o catálogo não sabe; o perfil do LangChain é consultado na criação do modelo.
    tool_calling: bool | None = None
    parallel_tool_calls: bool | None = None
    structured_output: StructuredOutput = "default"
    options: dict[str, Any] = Field(default_factory=dict)
    checked_on: date | None = None
    tested: Literal["mock", "live"] | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        if self.provider not in PROVIDERS:
            raise ValueError(f"provedor desconhecido no catálogo: {self.provider}")
        if self.provider in PAID_ONLY_PROVIDERS and self.is_free:
            raise ValueError(f"{self.model}: a API de {self.provider} não tem camada gratuita")
        # Defesa contra um catálogo mal editado: no OpenRouter, só estes IDs não cobram.
        if (
            self.provider == "openrouter"
            and self.cost == "free"
            and not (self.model == "openrouter/free" or self.model.endswith(":free"))
        ):
            raise ValueError(f"{self.model}: no OpenRouter, só openrouter/free e IDs ':free' são gratuitos")
        # Modelos "-cloud" do Ollama rodam nos servidores do Ollama, não na sua máquina.
        if self.cost == "local" and (self.provider != "ollama" or "cloud" in self.model):
            raise ValueError(f"{self.model}: apenas modelos do Ollama executados localmente são 'local'")
        return self

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def display_name(self) -> str:
        return self.label or self.model

    @property
    def is_free(self) -> bool:
        return self.cost in FREE_COSTS

    @property
    def supports_structured_output(self) -> bool:
        return self.structured_output != "none"

    @property
    def structured_output_method(self) -> str | None:
        """Método para with_structured_output; None usa o padrão da integração."""
        return None if self.structured_output in ("default", "none") else self.structured_output

    @property
    def thread_scope(self) -> str:
        """Prefixo das conversas deste modelo: históricos de modelos diferentes não se misturam."""
        return hashlib.sha1(self.key.encode()).hexdigest()[:8]


@lru_cache
def _load(path: Path) -> tuple[ModelSpec, ...]:
    with path.open("rb") as file:
        data = tomllib.load(file)
    specs = tuple(ModelSpec.model_validate(item) for item in data.get("models", []))
    keys = [spec.key for spec in specs]
    duplicated = {key for key in keys if keys.count(key) > 1}
    if duplicated:
        raise ValueError(f"modelos repetidos no catálogo: {sorted(duplicated)}")
    return specs


def load_catalog(settings: Settings) -> tuple[ModelSpec, ...]:
    return _load(Path(settings.llm_catalog_path or DEFAULT_CATALOG_PATH))


def find_model(settings: Settings, provider: str, model: str) -> ModelSpec | None:
    return next(
        (s for s in load_catalog(settings) if s.provider == provider and s.model == model), None
    )


def selectable_models(settings: Settings) -> list[ModelSpec]:
    """Modelos oferecidos na interface: no modo gratuito, os pagos nem aparecem."""
    catalog = load_catalog(settings)
    return [s for s in catalog if s.is_free or not settings.llm_free_only]


def _cost_policy_message(provider: str, model: str, spec: ModelSpec | None) -> str:
    if spec is not None and spec.cost == "paid":
        reason = f"'{model}' ({provider}) é um modelo pago"
    else:
        reason = (
            f"'{model}' ({provider}) não está no catálogo como gratuito, então seu custo não foi "
            "verificado"
        )
    return (
        f"Modelo bloqueado pelo modo gratuito (LLM_FREE_ONLY=true): {reason}. Escolha um modelo "
        "gratuito ou local do catálogo. Para liberar outro modelo, confirme na página de preços "
        "do provedor que ele é gratuito na sua conta e adicione-o a finance_copilot/llm/models.toml "
        "com cost = \"free\" (veja docs/LLM_PROVIDERS.md). Para usar modelos pagos de propósito, "
        "defina LLM_FREE_ONLY=false."
    )


def resolve_model(settings: Settings) -> ModelSpec:
    """Valida provedor, custo e compatibilidade, sem chamar a rede. Nunca escolhe outro modelo."""
    provider, model = settings.llm_provider, settings.llm_model.strip()
    if provider not in PROVIDERS:
        raise UnknownProviderError(
            f"Provedor de LLM desconhecido: '{provider}'. Use um destes em LLM_PROVIDER: "
            f"{', '.join(PROVIDERS)}."
        )
    if not model:
        raise IncompatibleModelError("LLM_MODEL está vazio: informe o modelo do provedor.")
    spec = find_model(settings, provider, model)
    if settings.llm_free_only and (spec is None or not spec.is_free):
        raise CostPolicyError(_cost_policy_message(provider, model, spec))
    if spec is None:
        spec = ModelSpec(provider=provider, model=model)
    if spec.tool_calling is False:
        raise IncompatibleModelError(
            f"O modelo '{model}' não suporta tool calling. O agente precisa de ferramentas para "
            "ler e alterar os dados; escolha outro modelo do catálogo."
        )
    return spec
