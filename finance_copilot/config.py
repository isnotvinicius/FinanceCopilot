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
PROVIDER_API_KEYS = {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}


def today_local() -> date:
    """Data de hoje no fuso da aplicação, independente do fuso do servidor."""
    return datetime.now(TIMEZONE).date()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float | None = None
    llm_timeout_seconds: int = 60
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "finance-copilot"

    finance_db_path: Path = Path("data/finance.db")
    agent_state_db_path: Path = Path("data/agent_state.db")

    @field_validator(
        "llm_temperature", "anthropic_api_key", "openai_api_key", "langsmith_api_key", mode="before"
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
        return self.llm_api_key is not None


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
