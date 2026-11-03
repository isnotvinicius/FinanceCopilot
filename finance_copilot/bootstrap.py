"""Monta os objetos da aplicação (composition root)."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.config import Settings, configure_tracing, get_settings
from finance_copilot.database import Database
from finance_copilot.llm import LLMConfigurationError, ModelSpec, create_chat_model
from finance_copilot.service import FinanceService


@dataclass
class App:
    settings: Settings
    service: FinanceService
    model: BaseChatModel | None
    session: AgentSession | None
    tracing_enabled: bool
    model_spec: ModelSpec | None = None
    # Por que o chat está inativo (sem chave, modelo pago no modo gratuito etc.).
    llm_error: str | None = None


def build_app(
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
    service: FinanceService | None = None,
) -> App:
    """Sem um modelo válido, o dashboard e os formulários continuam funcionando; só o chat fica
    inativo, com o motivo em `llm_error`. Nunca há troca automática para outro modelo."""
    settings = settings or get_settings()
    tracing_enabled = configure_tracing(settings)
    service = service or FinanceService(Database(settings.finance_db_path))
    spec, llm_error = None, None
    if model is None:
        try:
            bundle = create_chat_model(settings)
        except LLMConfigurationError as exc:
            llm_error = str(exc)
        else:
            model, spec = bundle.model, bundle.spec
    session = None
    if model is not None:
        graph = build_agent(model, open_checkpointer(settings.agent_state_db_path))
        session = AgentSession(graph, service)
    return App(settings, service, model, session, tracing_enabled, spec, llm_error)
