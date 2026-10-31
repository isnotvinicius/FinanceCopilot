"""Modelo de chat roteirizado para testes sem rede nem credenciais.

Ele NÃO testa a inteligência da LLM: devolve as respostas programadas. Serve para exercitar de
verdade o grafo, as tools, o middleware de aprovação, o checkpoint e o banco. A escolha de tools
por um modelo real é verificada em tests/test_llm_live.py.
"""

from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict


class ScriptedChatModel(BaseChatModel):
    """Cada item do roteiro é uma AIMessage, uma Exception a lançar ou uma função das mensagens."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[Any]
    calls: list[list[BaseMessage]] = []
    bound_tools: list[str] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        self.bound_tools[:] = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("O roteiro acabou: o agente chamou o modelo mais vezes que o esperado.")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        message = step(messages) if isinstance(step, Callable) else step
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def scripted(*steps) -> ScriptedChatModel:
    return ScriptedChatModel(script=list(steps), calls=[], bound_tools=[])


def call(name: str, call_id: str | None = None, text: str = "", **args) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{uuid4().hex}", "type": "tool_call"}],
    )


def calls(*items: tuple[str, dict]) -> AIMessage:
    """Várias tool calls na mesma resposta do modelo."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{uuid4().hex}", "type": "tool_call"}
            for name, args in items
        ],
    )


def reply(text: str) -> AIMessage:
    return AIMessage(content=text)
