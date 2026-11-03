"""Configuração da aplicação, lida de variáveis de ambiente e do arquivo .env."""

import logging
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

TIMEZONE = ZoneInfo("America/Sao_Paulo")
# Atributo de Settings com a chave de cada provedor. Cada provedor tem a sua variável: uma chave
# nunca é reaproveitada para outro serviço. O Ollama roda localmente e não usa chave.
PROVIDER_API_KEYS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "groq": "groq_api_key",
    "google_genai": "google_api_key",
    "openrouter": "openrouter_api_key",
}
KEYLESS_PROVIDERS = {"ollama"}


def today_local() -> date:
    """Data de hoje no fuso da aplicação, independente do fuso do servidor."""
    return datetime.now(TIMEZONE).date()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float | None = None
    llm_timeout_seconds: int = 60
    # Com true, só modelos marcados como gratuitos ou locais no catálogo podem ser usados.
    llm_free_only: bool = True
    # Catálogo alternativo; vazio usa finance_copilot/llm/models.toml.
    llm_catalog_path: Path | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "finance-copilot"

    finance_db_path: Path = Path("data/finance.db")
    agent_state_db_path: Path = Path("data/agent_state.db")

    @field_validator(
        "llm_temperature",
        "llm_catalog_path",
        "anthropic_api_key",
        "openai_api_key",
        "groq_api_key",
        "google_api_key",
        "openrouter_api_key",
        "langsmith_api_key",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value):
        # Linhas "CHAVE=" do .env chegam como string vazia.
        return None if value == "" else value

    @property
    def llm_api_key(self) -> str | None:
        attribute = PROVIDER_API_KEYS.get(self.llm_provider)
        return getattr(self, attribute) if attribute else None

    @property
    def llm_configured(self) -> bool:
        """Há credencial (ou ela não é necessária). Custo e compatibilidade são validados em llm/."""
        return self.llm_provider in KEYLESS_PROVIDERS or self.llm_api_key is not None

    def with_llm(self, provider: str, model: str) -> "Settings":
        """Cópia com outro provedor/modelo, usada pela seleção na interface. O .env não muda."""
        return self.model_copy(update={"llm_provider": provider, "llm_model": model})


def configure_tracing(settings: Settings) -> bool:
    """Ativa o LangSmith somente quando há flag e chave; retorna se ficou ativo.

    O LangChain lê estas variáveis diretamente de os.environ, então não há código de
    tracing no agente. Sem chave, desligamos a flag para evitar erros em toda execução.
    """
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    if settings.langsmith_tracing and not enabled:
        logger.warning("LANGSMITH_TRACING=true sem LANGSMITH_API_KEY: tracing desativado.")
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return enabled


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
