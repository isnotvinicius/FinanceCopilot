"""Fixtures comuns. Nenhum teste aqui acessa a rede nem exige credenciais."""

import os
from datetime import date

import pytest

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.database import Database
from finance_copilot.schemas import TransactionInput
from finance_copilot.service import FinanceService
from tests.fakes import scripted

os.environ["LANGSMITH_TRACING"] = "false"

TODAY = date(2026, 10, 15)


@pytest.fixture
def service(tmp_path) -> FinanceService:
    return FinanceService(Database(tmp_path / "finance.db"), clock=lambda: TODAY)


class Builder:
    """Cria transações de teste com chaves únicas, sem repetir campos irrelevantes."""

    def __init__(self, service: FinanceService) -> None:
        self.service = service
        self.counter = 0

    def add(self, cents, category="Alimentação", on=TODAY, kind="expense", description="Teste"):
        self.counter += 1
        return self.service.create_transaction(
            TransactionInput(
                kind=kind,
                amount_cents=cents,
                description=description,
                category=category,
                occurred_on=on,
            ),
            idempotency_key=f"test:{self.counter}",
            source="manual",
        ).transaction


@pytest.fixture
def builder(service) -> Builder:
    return Builder(service)


@pytest.fixture
def checkpointer(tmp_path):
    saver = open_checkpointer(tmp_path / "agent_state.db")
    yield saver
    saver.conn.close()


@pytest.fixture
def make_session(service, checkpointer):
    """Sessão do agente real com a LLM roteirizada (tests/fakes.py)."""
    def factory(*steps, **agent_kwargs):
        model = scripted(*steps)
        agent_kwargs.setdefault("model_retries", 0)
        session = AgentSession(build_agent(model, checkpointer, **agent_kwargs), service)
        session.model = model
        session.graph = session._graph
        return session

    return factory
