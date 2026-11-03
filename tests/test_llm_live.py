"""Avaliação com o modelo real: confere se a LLM escolhe as tools certas e não inventa valores.

Sem um modelo válido no .env (chave ausente, modelo bloqueado por LLM_FREE_ONLY etc.), estes
testes são pulados com o motivo. Para rodar (consome cota ou créditos do provedor configurado):
    .venv/bin/python -m pytest -m live -v

Em planos gratuitos com limite por minuto (Groq: 8K tokens/min), prefira scripts/smoke_llm.py,
que faz poucas chamadas e espera entre elas.
"""

from datetime import date

import pytest

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession
from finance_copilot.config import Settings
from finance_copilot.database import Database
from finance_copilot.llm import LLMConfigurationError, build_chat_model
from finance_copilot.schemas import TransactionFilter
from finance_copilot.seed import seed
from finance_copilot.service import FinanceService

settings = Settings()


def _configuration_problem() -> str | None:
    try:
        build_chat_model(settings)
    except LLMConfigurationError as exc:
        return str(exc)
    return None


SKIP_REASON = _configuration_problem()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(SKIP_REASON is not None, reason=f"LLM indisponível: {SKIP_REASON}"),
]

TODAY = date(2026, 10, 15)


@pytest.fixture
def session(tmp_path):
    service = FinanceService(Database(tmp_path / "finance.db"), clock=lambda: TODAY)
    seed(service)
    saver = open_checkpointer(tmp_path / "state.db")
    agent_session = AgentSession(build_agent(build_chat_model(settings), saver), service)
    agent_session.service = service
    yield agent_session
    saver.conn.close()


def used_tools(session, thread) -> set[str]:
    return {event.name for event in session.history(thread)[-1].tool_events}


@pytest.mark.parametrize(
    ("question", "acceptable_tools"),
    [
        ("Quanto dinheiro eu tenho disponível neste mês?", {"get_financial_summary"}),
        ("Qual categoria representa minha maior despesa neste mês?", {"get_category_summary"}),
        ("Gastei mais com alimentação neste mês ou no mês passado?", {"compare_periods", "get_category_summary"}),
        ("Quanto economizaria reduzindo o lazer em 20%?", {"simulate_budget"}),
        ("Quais categorias mais contribuíram para o aumento dos meus gastos?", {"compare_periods"}),
        ("Se eu economizar R$ 300 por mês, quanto terei acumulado em seis meses?", {"simulate_budget"}),
    ],
)
def test_question_uses_appropriate_tool(session, question, acceptable_tools):
    assert session.send("t", question).error is None
    assert used_tools(session, "t") & acceptable_tools


def test_expense_message_proposes_write_and_waits_for_approval(session):
    before = session.service.list_transactions(TransactionFilter(limit=500)).total_count

    session.send("t", "Gastei 85 reais com pizza ontem")

    pending = session.pending_approval("t")
    assert pending is not None
    action = pending.actions[0]
    assert action.name == "create_transaction"
    assert action.args["occurred_on"] == "2026-10-14"
    assert action.args["category"] == "Alimentação"
    assert session.service.list_transactions(TransactionFilter(limit=500)).total_count == before


def test_followup_keeps_context(session):
    session.send("t", "Quanto gastei com alimentação em setembro?")
    session.send("t", "E em outubro?")
    events = session.history("t")[-1].tool_events
    assert events
    assert "Alimenta" in str(events[0].args) or "2026-10" in str(events[0].args)


def test_period_without_data_is_not_invented(session):
    session.send("t", "Quanto gastei com lazer em janeiro de 2020?")
    text = session.history("t")[-1].text.lower()
    assert "r$ 0" not in text
