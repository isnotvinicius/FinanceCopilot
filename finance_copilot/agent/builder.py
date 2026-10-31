"""Montagem do agente com create_agent.

O create_agent compila um grafo LangGraph com o ciclo modelo -> tools -> modelo. Não há grafo
próprio: um único agente com middlewares resolve o problema. Cada middleware existe por um motivo:

- dynamic_prompt: data de hoje e categorias precisam estar atualizadas em toda chamada, mesmo em
  conversas retomadas dias depois.
- HumanInTheLoopMiddleware: pausa o grafo antes de qualquer escrita.
- ModelCallLimitMiddleware: limita custo e impede laços infinitos de tool calling.
- ModelRetryMiddleware: tolera falhas transitórias do provedor.
- ToolErrorMiddleware: devolve erros de tool ao modelo, que pode corrigir ou explicar.
"""

import logging
import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from finance_copilot.agent.approvals import build_interrupt_on
from finance_copilot.agent.context import AgentContext
from finance_copilot.agent.prompts import render_system_prompt
from finance_copilot.agent.tools import ALL_TOOLS
from finance_copilot.errors import FinanceError

logger = logging.getLogger(__name__)

MODEL_CALLS_PER_RUN = 10

MODEL_FAILURE_MESSAGE = (
    "Não consegui falar com o modelo de IA agora, mesmo depois de tentar novamente. "
    "Nenhuma operação foi realizada. Tente de novo em instantes."
)


def open_checkpointer(path: str | Path) -> SqliteSaver:
    """Checkpointer em arquivo próprio, separado dos dados financeiros.

    O Streamlit executa cada sessão em uma thread diferente. O SqliteSaver protege a conexão com
    um lock, então uma conexão compartilhada é segura com check_same_thread=False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@dynamic_prompt
def _finance_system_prompt(request: ModelRequest) -> str:
    service = request.runtime.context.service
    return render_system_prompt(service.today(), service.list_categories())


def _is_transient(exc: Exception) -> bool:
    """Erros como chave inválida (401) ou pedido malformado (400) não melhoram com nova tentativa.

    Esses erros seguem até AgentSession, que orienta o usuário a revisar a configuração.
    """
    status = getattr(exc, "status_code", None)
    if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
        return False
    return getattr(exc, "is_retryable", True)


def _tool_error_message(exc: Exception, request) -> str:
    """Traduz exceções de tools em texto para o modelo, sem expor detalhes internos."""
    if isinstance(exc, FinanceError):
        return f"Erro: {exc}"
    if isinstance(exc, ValidationError):
        return f"Erro de validação: {exc.errors(include_url=False, include_input=False)}"
    if isinstance(exc, sqlite3.Error):
        logger.exception("Erro de banco na tool %s", request.tool_call["name"])
        return "Erro: falha ao acessar o banco de dados. A operação não foi concluída."
    logger.exception("Erro inesperado na tool %s", request.tool_call["name"])
    return "Erro interno ao executar a ferramenta. A operação não foi concluída."


def build_agent(
    model: BaseChatModel,
    checkpointer,
    *,
    model_retries: int = 2,
    model_calls_per_run: int = MODEL_CALLS_PER_RUN,
):
    return create_agent(
        model,
        tools=ALL_TOOLS,
        context_schema=AgentContext,
        checkpointer=checkpointer,
        middleware=[
            _finance_system_prompt,
            HumanInTheLoopMiddleware(interrupt_on=build_interrupt_on()),
            ModelCallLimitMiddleware(run_limit=model_calls_per_run, exit_behavior="end"),
            ModelRetryMiddleware(
                max_retries=model_retries,
                retry_on=_is_transient,
                on_failure=lambda exc: MODEL_FAILURE_MESSAGE,
            ),
            ToolErrorMiddleware(on_error=_tool_error_message),
        ],
        name="finance_copilot",
    )
