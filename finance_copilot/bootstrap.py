"""Monta os objetos da aplicação (composition root)."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.config import Settings, configure_tracing, get_settings
from finance_copilot.database import Database
from finance_copilot.llm import build_chat_model
from finance_copilot.service import FinanceService


@dataclass
class App:
    settings: Settings
    service: FinanceService
    model: BaseChatModel | None
    session: AgentSession | None
    tracing_enabled: bool


def build_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> App:
    """Sem chave de LLM, o dashboard e os formulários continuam funcionando; só o chat fica inativo."""
    settings = settings or get_settings()
    tracing_enabled = configure_tracing(settings)
    service = FinanceService(Database(settings.finance_db_path))
    if model is None and settings.llm_configured:
        model = build_chat_model(settings)
    session = None
    if model is not None:
        graph = build_agent(model, open_checkpointer(settings.agent_state_db_path))
        session = AgentSession(graph, service)
    return App(settings, service, model, session, tracing_enabled)
