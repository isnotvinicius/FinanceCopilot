"""Fronteira entre a interface e o grafo do agente.

A interface só conhece esta classe: envia mensagens, consulta o histórico e responde às aprovações.
O estado vem sempre do checkpoint, portanto recarregar a página não perde a conversa nem a
aprovação pendente.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from finance_copilot.agent.context import AgentContext
from finance_copilot.llm.errors import classify_llm_error
from finance_copilot.service import FinanceService

logger = logging.getLogger(__name__)

AGENT_FAILURE_MESSAGE = (
    "Não foi possível concluir o pedido por uma falha na comunicação com o modelo de IA. "
    "Verifique a configuração (.env) e tente novamente."
)
# Texto gerado pelo HumanInTheLoopMiddleware ao rejeitar; um teste fixa este contrato.
_REJECTION_PREFIX = "User rejected the tool call"
# Texto gerado pelo ModelCallLimitMiddleware ao atingir o limite.
_CALL_LIMIT_PREFIX = "Model call limits exceeded"
CALL_LIMIT_MESSAGE = (
    "Este pedido exigiu etapas demais e foi interrompido por segurança. "
    "Tente dividi-lo em perguntas menores."
)


class PendingApprovalError(RuntimeError):
    """Há uma aprovação pendente: a conversa não pode avançar antes de respondê-la."""


class StaleApprovalError(RuntimeError):
    """A aprovação respondida não é a que está pendente (duplo clique, outra aba, reenvio)."""


@dataclass
class ToolEvent:
    """Evento observável de uma execução: nunca inclui o raciocínio interno do modelo."""

    name: str
    args: dict[str, Any]
    status: Literal["ok", "error", "rejected", "pending"]
    result: Any = None


@dataclass
class ChatEntry:
    role: Literal["user", "assistant"]
    text: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)

    @property
    def has_simulation(self) -> bool:
        """Vem do resultado da tool, não do texto do modelo."""
        return any(
            isinstance(event.result, dict) and event.result.get("is_simulation")
            for event in self.tool_events
        )


@dataclass
class PendingAction:
    name: str
    args: dict[str, Any]
    description: str


@dataclass
class PendingApproval:
    interrupt_id: str
    actions: list[PendingAction]


@dataclass
class TurnResult:
    error: str | None = None


def _parse_result(content: Any) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def _event_for(call: dict, results: dict[str, ToolMessage]) -> ToolEvent:
    message = results.get(call["id"])
    if message is None:
        return ToolEvent(call["name"], call["args"], "pending")
    result = _parse_result(message.content)
    if message.status != "error":
        return ToolEvent(call["name"], call["args"], "ok", result)
    rejected = str(message.content).startswith(_REJECTION_PREFIX)
    return ToolEvent(call["name"], call["args"], "rejected" if rejected else "error", result)


def messages_to_entries(messages: list[BaseMessage]) -> list[ChatEntry]:
    """Agrupa as mensagens em turnos: cada resposta reúne seus textos e chamadas de tools."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    entries: list[ChatEntry] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            entries.append(ChatEntry("user", message.text))
        elif isinstance(message, AIMessage):
            if not entries or entries[-1].role != "assistant":
                entries.append(ChatEntry("assistant"))
            entry = entries[-1]
            text = message.text
            if text.startswith(_CALL_LIMIT_PREFIX):
                text = CALL_LIMIT_MESSAGE
            if text:
                entry.text = f"{entry.text}\n\n{text}".strip()
            entry.tool_events += [_event_for(call, results) for call in message.tool_calls]
    return entries


class AgentSession:
    def __init__(self, graph, service: FinanceService) -> None:
        self._graph = graph
        self._context = AgentContext(service=service)

    @staticmethod
    def _config(thread_id: str) -> dict:
        # tags e metadata aparecem no LangSmith quando o tracing está ativo.
        return {
            "configurable": {"thread_id": thread_id},
            "run_name": "finance_copilot_turn",
            "tags": ["finance-copilot"],
            "metadata": {"thread_id": thread_id},
        }

    def _state(self, thread_id: str):
        return self._graph.get_state(self._config(thread_id))

    def history(self, thread_id: str) -> list[ChatEntry]:
        return messages_to_entries(self._state(thread_id).values.get("messages", []))

    def pending_approval(self, thread_id: str) -> PendingApproval | None:
        interrupts = self._state(thread_id).interrupts
        if not interrupts:
            return None
        interrupt = interrupts[0]
        actions = [
            PendingAction(action["name"], action["args"], action.get("description", ""))
            for action in interrupt.value["action_requests"]
        ]
        return PendingApproval(interrupt.id, actions)

    def send(self, thread_id: str, text: str) -> TurnResult:
        """Executa um turno. Sem a guarda, uma nova mensagem deixaria a tool call anterior sem
        resposta no histórico, e o provedor recusaria todas as chamadas seguintes."""
        if self.pending_approval(thread_id):
            raise PendingApprovalError("Responda à confirmação pendente antes de continuar.")
        return self._run({"messages": [HumanMessage(text)]}, thread_id)

    def decide(
        self, thread_id: str, interrupt_id: str, *, approve: bool, reason: str | None = None
    ) -> TurnResult:
        """Retoma a execução pausada, aplicando a decisão a todas as ações do cartão.

        Só aceita a aprovação que está pendente de fato: um segundo clique cai em
        StaleApprovalError e não retoma nada. Se duas retomadas passarem juntas por aqui, a
        chave de idempotência no banco ainda impede a gravação em dobro.
        """
        pending = self.pending_approval(thread_id)
        if pending is None or pending.interrupt_id != interrupt_id:
            raise StaleApprovalError("Esta confirmação já foi respondida.")
        if approve:
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject"}
            if reason:
                decision["message"] = reason
        decisions = [decision for _ in pending.actions]
        return self._run(Command(resume={"decisions": decisions}), thread_id)

    def _run(self, graph_input: Any, thread_id: str) -> TurnResult:
        try:
            self._graph.invoke(
                graph_input, self._config(thread_id), context=self._context, version="v2"
            )
        except Exception as exc:
            logger.exception("Falha ao executar o agente na conversa %s", thread_id)
            return TurnResult(error=classify_llm_error(exc).message or AGENT_FAILURE_MESSAGE)
        return TurnResult()
