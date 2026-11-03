"""Montagem do agente com create_agent.

O create_agent compila um grafo LangGraph com o ciclo modelo -> tools -> modelo. Não há grafo
próprio: um único agente com middlewares resolve o problema. Cada middleware existe por um motivo:

- dynamic_prompt: data de hoje e categorias precisam estar atualizadas em toda chamada, mesmo em
  conversas retomadas dias depois.
- HumanInTheLoopMiddleware: pausa o grafo antes de qualquer escrita.
- ModelCallLimitMiddleware: limita custo e impede laços infinitos de tool calling.
- ModelRetryMiddleware: tolera falhas transitórias do provedor, com limite de tentativas. Cota
  esgotada, falta de créditos e esperas longas de rate limit não são repetidas (llm/errors.py).
- ToolErrorMiddleware: devolve erros de tool ao modelo, que pode corrigir ou explicar.
- _unique_tool_call_ids: garante ids de tool call únicos na conversa, qualquer que seja o provedor.
- _recover_rejected_tool_calls: quando o provedor recusa uma tool call inválida (Groq), o erro volta
  ao modelo pelo mesmo caminho dos demais provedores, para que ele corrija os argumentos.
"""

import logging
import sqlite3
from pathlib import Path
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    after_model,
    dynamic_prompt,
    wrap_model_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from finance_copilot.agent.approvals import build_interrupt_on
from finance_copilot.agent.context import AgentContext
from finance_copilot.agent.prompts import render_system_prompt
from finance_copilot.agent.tools import ALL_TOOLS
from finance_copilot.errors import FinanceError
from finance_copilot.llm.errors import classify_llm_error, recover_rejected_tool_calls

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


@after_model
def _unique_tool_call_ids(state, runtime) -> dict | None:
    """O id da tool call compõe a chave de idempotência das escritas (tools.py::_idempotency_key).

    Anthropic e OpenAI geram ids únicos, mas alguns backends abertos devolvem ids vazios ou
    repetidos ("call_0" em todo turno). Sem esta correção, uma segunda despesa legítima seria
    tratada como replay e não seria gravada. Roda antes da aprovação humana e fica salva no
    checkpoint, então o id continua estável em retomadas.
    """
    messages = state["messages"]
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return None
    used = {c["id"] for m in messages[:-1] if isinstance(m, AIMessage) for c in m.tool_calls}
    fixed = []
    for tool_call in last.tool_calls:
        if not tool_call.get("id") or tool_call["id"] in used:
            tool_call = {**tool_call, "id": f"call_{uuid4().hex}"}
        used.add(tool_call["id"])
        fixed.append(tool_call)
    if fixed == last.tool_calls:
        return None
    logger.warning("Ids de tool call ausentes ou repetidos foram substituídos.")
    # Mesmo id de mensagem: o reducer substitui a AIMessage em vez de acrescentar outra.
    return {"messages": [last.model_copy(update={"tool_calls": fixed})]}


@wrap_model_call
def _recover_rejected_tool_calls(request, handler):
    """Converte a recusa do provedor em uma AIMessage com a tool call que o modelo gerou.

    Os argumentos inválidos não pedem aprovação (approvals.py::_has_valid_args) e são recusados
    pelo schema da tool, sem gravar nada. O ToolErrorMiddleware devolve o erro ao modelo, que
    corrige. O ModelCallLimitMiddleware limita as tentativas.
    """
    try:
        return handler(request)
    except Exception as exc:
        tool_calls = recover_rejected_tool_calls(exc)
        if tool_calls is None:
            raise
        logger.warning("Tool call recusada pelo provedor; erro devolvido ao modelo: %s", exc)
        return AIMessage(content="", tool_calls=tool_calls)


def _is_transient(exc: Exception) -> bool:
    """Erros como chave inválida (401), pedido malformado (400), cota diária esgotada ou falta de
    créditos (402) não melhoram com nova tentativa: repetir só gastaria requisições da cota.

    Esses erros seguem até AgentSession, que mostra ao usuário o motivo e o que fazer.
    """
    return classify_llm_error(exc).retryable


def _model_failure_message(exc: Exception) -> str:
    """Texto salvo na conversa quando as tentativas acabam (ex.: rate limit persistente)."""
    return classify_llm_error(exc).message or MODEL_FAILURE_MESSAGE


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
                on_failure=_model_failure_message,
                # 2 s e 4 s: o suficiente para o limite por minuto de um plano gratuito aliviar.
                initial_delay=2.0,
            ),
            # Depois do retry na lista, ou seja, mais perto do modelo: a recusa não é repetida.
            _recover_rejected_tool_calls,
            ToolErrorMiddleware(on_error=_tool_error_message),
            # Último da lista: seu after_model roda primeiro, antes do HumanInTheLoopMiddleware.
            _unique_tool_call_ids,
        ],
        name="finance_copilot",
    )
