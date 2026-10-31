"""Testes do agente com o grafo real, checkpoint SQLite real e banco real; só a LLM é roteirizada."""

import json
import sqlite3
from datetime import date

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from finance_copilot.agent.approvals import build_interrupt_on
from finance_copilot.agent.builder import MODEL_FAILURE_MESSAGE, build_agent, open_checkpointer
from finance_copilot.agent.context import AgentContext
from finance_copilot.agent.extraction import extract_transaction_draft
from finance_copilot.agent.session import (
    AGENT_FAILURE_MESSAGE,
    CALL_LIMIT_MESSAGE,
    AgentSession,
    PendingApprovalError,
    StaleApprovalError,
)
from finance_copilot.agent.tool_schemas import TransactionDraft
from finance_copilot.agent.tools import ALL_TOOLS, READ_TOOLS, WRITE_TOOLS
from finance_copilot.schemas import TransactionFilter
from tests.fakes import call, calls, reply, scripted

THREAD = "thread-1"
PIZZA = {
    "kind": "expense",
    "amount": 85,
    "description": "Pizza",
    "category": "Alimentação",
    "occurred_on": "2026-10-14",
}


@pytest.fixture
def checkpointer(tmp_path):
    saver = open_checkpointer(tmp_path / "agent_state.db")
    yield saver
    saver.conn.close()


@pytest.fixture
def make_session(service, checkpointer):
    def factory(*steps, **agent_kwargs):
        model = scripted(*steps)
        agent_kwargs.setdefault("model_retries", 0)
        session = AgentSession(build_agent(model, checkpointer, **agent_kwargs), service)
        session.model = model
        session.graph = session._graph
        return session

    return factory


def transaction_count(service) -> int:
    return service.list_transactions(TransactionFilter(limit=500)).total_count


def tool_messages(model_call) -> list[ToolMessage]:
    return [m for m in model_call if isinstance(m, ToolMessage)]


# --- Definição das tools --------------------------------------------------------------------


def test_every_required_tool_is_exposed(make_session):
    session = make_session(reply("oi"))
    session.send(THREAD, "oi")
    assert set(session.model.bound_tools) == {
        "create_transaction",
        "get_transactions",
        "get_financial_summary",
        "get_category_summary",
        "compare_periods",
        "simulate_budget",
        "set_monthly_salary",
        "update_transaction",
        "delete_transaction",
    }


def test_all_write_tools_require_approval_and_read_tools_do_not():
    gated = set(build_interrupt_on())
    assert gated == {tool.name for tool in WRITE_TOOLS}
    assert gated.isdisjoint(tool.name for tool in READ_TOOLS)
    # Regra para quem criar uma tool nova: precisa ser classificada em uma das listas.
    assert len(ALL_TOOLS) == len(WRITE_TOOLS) + len(READ_TOOLS)


def test_tool_schema_hides_runtime_and_describes_arguments():
    for tool in ALL_TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        assert "runtime" not in schema.get("properties", {})
        assert tool.description
    amount = {t.name: t for t in ALL_TOOLS}["create_transaction"].tool_call_schema.model_json_schema()
    assert amount["properties"]["amount"]["type"] == "number"


def test_no_tool_accepts_raw_sql():
    for tool in ALL_TOOLS:
        names = tool.tool_call_schema.model_json_schema().get("properties", {})
        assert not {"sql", "query", "statement"} & set(names)


def test_system_prompt_is_rendered_with_today_and_categories(make_session):
    session = make_session(reply("oi"))
    session.send(THREAD, "oi")
    system = session.model.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "15/10/2026" in system.text
    assert "quinta-feira" in system.text
    assert "Alimentação" in system.text
    assert "Nunca invente valores" in system.text


# --- Seleção e execução de tools ------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_key"),
    [
        ("get_financial_summary", {"start_month": "2026-10"}, "balance"),
        ("get_category_summary", {"start_month": "2026-10"}, "top_category"),
        ("get_transactions", {"category": "Alimentação"}, "items"),
        ("compare_periods", {"baseline_start": "2026-09", "target_start": "2026-10"}, "comparable"),
        ("simulate_budget", {"extra_monthly_saving": 300, "months": 6}, "is_simulation"),
    ],
)
def test_read_tools_run_without_approval_and_feed_the_model(
    make_session, builder, tool_name, args, expected_key
):
    builder.add(5000, on=date(2026, 10, 1))
    session = make_session(call(tool_name, **args), reply("Pronto."))

    result = session.send(THREAD, "pergunta")

    assert result.error is None
    assert session.pending_approval(THREAD) is None
    entry = session.history(THREAD)[-1]
    assert entry.text == "Pronto."
    assert [e.name for e in entry.tool_events] == [tool_name]
    assert entry.tool_events[0].status == "ok"
    assert expected_key in entry.tool_events[0].result
    # O modelo recebeu o resultado da tool antes de escrever a resposta final.
    assert tool_messages(session.model.calls[1])[0].name == tool_name


def test_tool_results_are_real_database_values(make_session, builder, service):
    service.set_monthly_salary(500000, effective_from="2026-10", idempotency_key="s")
    builder.add(30000, "Lazer", date(2026, 10, 1))
    session = make_session(call("get_financial_summary", start_month="2026-10"), reply("ok"))

    session.send(THREAD, "Quanto tenho disponível?")

    result = session.history(THREAD)[-1].tool_events[0].result
    assert result["balance"] == {"cents": 470000, "brl": "R$ 4.700,00"}


def test_agent_can_chain_tools_before_answering(make_session, builder):
    builder.add(5000, on=date(2026, 10, 1))
    session = make_session(
        call("get_category_summary", start_month="2026-10"),
        call("get_financial_summary", start_month="2026-10"),
        reply("Resposta final"),
    )

    session.send(THREAD, "Análise")

    events = session.history(THREAD)[-1].tool_events
    assert [e.name for e in events] == ["get_category_summary", "get_financial_summary"]


def test_answer_without_tools_is_possible(make_session):
    session = make_session(reply("Em que posso ajudar?"))
    session.send(THREAD, "oi")
    assert session.history(THREAD)[-1].tool_events == []


def test_simulation_result_is_flagged_for_the_interface(make_session):
    session = make_session(
        call("simulate_budget", extra_monthly_saving=300, months=6), reply("Simulação: R$ 1.800,00")
    )
    session.send(THREAD, "Se eu economizar 300 por mês?")
    entry = session.history(THREAD)[-1]
    assert entry.has_simulation is True
    assert entry.tool_events[0].result["accumulated_saving"]["brl"] == "R$ 1.800,00"


def test_real_tool_answer_is_not_flagged_as_simulation(make_session):
    session = make_session(call("get_financial_summary", start_month="2026-10"), reply("ok"))
    session.send(THREAD, "saldo")
    assert session.history(THREAD)[-1].has_simulation is False


# --- Aprovação humana -----------------------------------------------------------------------


def test_write_tool_pauses_before_touching_the_database(make_session, service):
    session = make_session(call("create_transaction", **PIZZA))

    session.send(THREAD, "Gastei 85 reais com pizza ontem")

    assert transaction_count(service) == 0
    pending = session.pending_approval(THREAD)
    assert [a.name for a in pending.actions] == ["create_transaction"]
    assert "R$ 85,00" in pending.actions[0].description
    assert "14/10/2026" in pending.actions[0].description
    assert session.history(THREAD)[-1].tool_events[0].status == "pending"


def test_approval_persists_and_agent_reports_real_result(make_session, service):
    session = make_session(call("create_transaction", **PIZZA), reply("Registrado!"))
    session.send(THREAD, "Gastei 85 reais com pizza ontem")
    pending = session.pending_approval(THREAD)

    result = session.decide(THREAD, pending.interrupt_id, approve=True)

    assert result.error is None
    assert session.pending_approval(THREAD) is None
    items = service.list_transactions(TransactionFilter()).items
    assert [(t.description, t.amount.cents, t.source) for t in items] == [("Pizza", 8500, "agent")]
    entry = session.history(THREAD)[-1]
    assert entry.text == "Registrado!"
    assert entry.tool_events[0].status == "ok"
    assert entry.tool_events[0].result["transaction"]["amount"]["brl"] == "R$ 85,00"


def test_rejection_does_not_write_and_reason_reaches_the_model(make_session, service):
    session = make_session(call("create_transaction", **PIZZA), reply("Ok, não registrei."))
    session.send(THREAD, "Gastei 85")
    pending = session.pending_approval(THREAD)

    session.decide(THREAD, pending.interrupt_id, approve=False, reason="foi 80, não 85")

    assert transaction_count(service) == 0
    entry = session.history(THREAD)[-1]
    assert entry.tool_events[0].status == "rejected"
    assert "foi 80, não 85" in tool_messages(session.model.calls[-1])[0].text


def test_rejection_without_reason_is_marked_as_rejected(make_session):
    session = make_session(call("create_transaction", **PIZZA), reply("Ok."))
    session.send(THREAD, "x")
    session.decide(THREAD, session.pending_approval(THREAD).interrupt_id, approve=False)
    assert session.history(THREAD)[-1].tool_events[0].status == "rejected"


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_text"),
    [
        ("update_transaction", {"amount": 99}, "R$ 50,00 → **R$ 99,00**"),
        ("delete_transaction", {}, "Excluir transação"),
    ],
)
def test_update_and_delete_require_approval_and_preview_database_values(
    make_session, service, builder, tool_name, args, expected_text
):
    created = builder.add(5000, description="Original")
    session = make_session(call(tool_name, transaction_id=created.id, **args), reply("Feito."))

    session.send(THREAD, "Altere")

    pending = session.pending_approval(THREAD)
    assert expected_text in pending.actions[0].description
    assert "Original" in pending.actions[0].description
    assert service.get_transaction(created.id).amount.cents == 5000

    session.decide(THREAD, pending.interrupt_id, approve=True)

    total = service.list_transactions(TransactionFilter()).total_expenses.cents
    assert total == (9900 if tool_name == "update_transaction" else 0)


def test_set_salary_requires_approval(make_session, service):
    session = make_session(call("set_monthly_salary", amount=7000), reply("Salário definido."))

    session.send(THREAD, "Meu salário é 7000")

    assert service.get_salary("2026-10") is None
    session.decide(THREAD, session.pending_approval(THREAD).interrupt_id, approve=True)
    assert service.get_salary("2026-10").cents == 700000
    assert transaction_count(service) == 0


def test_multiple_writes_in_one_turn_go_to_the_same_card(make_session, service):
    session = make_session(
        calls(
            ("create_transaction", PIZZA),
            ("create_transaction", {**PIZZA, "description": "Refrigerante", "amount": 10}),
        ),
        reply("Dois registros."),
    )
    session.send(THREAD, "Pizza e refrigerante")

    pending = session.pending_approval(THREAD)
    assert len(pending.actions) == 2

    session.decide(THREAD, pending.interrupt_id, approve=True)
    assert transaction_count(service) == 2


def test_new_message_is_refused_while_approval_is_pending(make_session, service):
    session = make_session(call("create_transaction", **PIZZA))
    session.send(THREAD, "Gastei 85")
    with pytest.raises(PendingApprovalError):
        session.send(THREAD, "outra coisa")
    assert session.pending_approval(THREAD) is not None


def test_preview_warns_about_identical_transaction(make_session, service, builder):
    builder.add(8500, description="pizza", on=date(2026, 10, 14))
    session = make_session(call("create_transaction", **PIZZA))
    session.send(THREAD, "Gastei 85")
    assert "Já existe um lançamento igual" in session.pending_approval(THREAD).actions[0].description


def test_preview_warns_when_operation_will_fail(make_session, builder):
    session = make_session(call("delete_transaction", transaction_id=999))
    session.send(THREAD, "Apague")
    assert "será recusada" in session.pending_approval(THREAD).actions[0].description


# --- Prevenção de duplicação ----------------------------------------------------------------


def test_second_click_on_same_approval_is_refused(make_session, service):
    session = make_session(call("create_transaction", **PIZZA), reply("Registrado!"))
    session.send(THREAD, "Gastei 85")
    interrupt_id = session.pending_approval(THREAD).interrupt_id

    session.decide(THREAD, interrupt_id, approve=True)

    with pytest.raises(StaleApprovalError):
        session.decide(THREAD, interrupt_id, approve=True)
    assert transaction_count(service) == 1


def test_stale_interrupt_id_cannot_approve_a_newer_operation(make_session, service):
    session = make_session(
        call("create_transaction", **PIZZA),
        reply("Ok."),
        call("delete_transaction", transaction_id=1),
    )
    session.send(THREAD, "Gastei 85")
    first = session.pending_approval(THREAD).interrupt_id
    session.decide(THREAD, first, approve=True)
    session.send(THREAD, "Apague")

    with pytest.raises(StaleApprovalError):
        session.decide(THREAD, first, approve=True)
    assert transaction_count(service) == 1
    assert session.pending_approval(THREAD) is not None


def test_replaying_checkpoint_after_commit_does_not_duplicate(make_session, service):
    """Cenário de falha: a tool gravou, mas a execução é repetida a partir do checkpoint anterior.

    Reproduzimos com time travel do LangGraph, passando por cima da guarda da sessão, para provar
    que a barreira do banco funciona sozinha.
    """
    session = make_session(call("create_transaction", "call_fixed", **PIZZA), reply("1"), reply("2"))
    session.send(THREAD, "Gastei 85")
    config = {"configurable": {"thread_id": THREAD}}
    paused = session.graph.get_state(config).config
    context = AgentContext(service=service)
    resume = Command(resume={"decisions": [{"type": "approve"}]})

    session.graph.invoke(resume, config, context=context, version="v2")
    assert transaction_count(service) == 1

    session.graph.invoke(resume, paused, context=context, version="v2")

    assert transaction_count(service) == 1
    replayed = json.loads(tool_messages(session.model.calls[-1])[-1].text)
    assert replayed["replayed"] is True


def test_same_tool_call_id_in_different_threads_are_separate_operations(make_session, service):
    session = make_session(
        call("create_transaction", "call_same", **PIZZA),
        reply("ok"),
        call("create_transaction", "call_same", **PIZZA),
        reply("ok"),
    )
    for thread in ("a", "b"):
        session.send(thread, "Gastei 85")
        session.decide(thread, session.pending_approval(thread).interrupt_id, approve=True)
    assert transaction_count(service) == 2


# --- Erros ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_args",
    [{"amount": -5}, {"amount": 0}, {"amount": "abc"}, {"amount": 10.123}, {"occurred_on": "ontem"}],
)
def test_invalid_arguments_skip_approval_and_return_to_model(make_session, service, bad_args):
    session = make_session(call("create_transaction", **{**PIZZA, **bad_args}), reply("Valor inválido."))

    session.send(THREAD, "Gastei")

    assert session.pending_approval(THREAD) is None
    assert transaction_count(service) == 0
    event = session.history(THREAD)[-1].tool_events[0]
    assert event.status == "error"
    assert session.history(THREAD)[-1].text == "Valor inválido."


def test_business_error_after_approval_returns_to_model(make_session, service):
    session = make_session(
        call("create_transaction", **{**PIZZA, "category": "Pets"}), reply("Essa categoria não existe.")
    )
    session.send(THREAD, "Gastei")
    pending = session.pending_approval(THREAD)
    assert "será recusada" in pending.actions[0].description

    session.decide(THREAD, pending.interrupt_id, approve=True)

    assert transaction_count(service) == 0
    event = session.history(THREAD)[-1].tool_events[0]
    assert event.status == "error"
    assert "Opções válidas" in event.result


def test_model_can_correct_itself_after_tool_error(make_session, service):
    session = make_session(
        call("get_category_summary", start_month="2026-13"),
        call("get_category_summary", start_month="2026-10"),
        reply("Corrigido."),
    )
    session.send(THREAD, "Categorias")
    assert [e.status for e in session.history(THREAD)[-1].tool_events] == ["error", "ok"]


def test_database_failure_is_reported_to_model_without_leaking_details(make_session, service, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error at /secret/path")

    monkeypatch.setattr(service, "financial_summary", broken)
    session = make_session(call("get_financial_summary", start_month="2026-10"), reply("Falha no banco."))

    session.send(THREAD, "saldo")

    event = session.history(THREAD)[-1].tool_events[0]
    assert event.status == "error"
    assert "/secret/path" not in event.result


def test_llm_failure_after_retries_gives_friendly_message_and_keeps_thread_usable(make_session):
    session = make_session(RuntimeError("HTTP 500 com detalhes internos"), reply("Voltei."))

    assert session.send(THREAD, "oi").error is None
    assert session.history(THREAD)[-1].text == MODEL_FAILURE_MESSAGE

    session.send(THREAD, "tudo bem?")
    assert session.history(THREAD)[-1].text == "Voltei."


def test_transient_llm_failure_is_retried(service, checkpointer, make_session, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    session = make_session(ConnectionError("instável"), reply("Funcionou."), model_retries=1)
    session.send(THREAD, "oi")
    assert session.history(THREAD)[-1].text == "Funcionou."


def test_invalid_api_key_is_not_retried_and_points_to_configuration(make_session):
    response = httpx.Response(401, request=httpx.Request("POST", "http://local"))
    auth_error = anthropic.AuthenticationError("chave inválida", response=response, body=None)
    session = make_session(auth_error, reply("Voltei."), model_retries=2)

    result = session.send(THREAD, "oi")

    assert result.error == AGENT_FAILURE_MESSAGE
    assert len(session.model.calls) == 1
    session.send(THREAD, "de novo")
    assert session.history(THREAD)[-1].text == "Voltei."


def test_rate_limit_is_retried(make_session, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    response = httpx.Response(429, request=httpx.Request("POST", "http://local"))
    limited = anthropic.RateLimitError("limite", response=response, body=None)
    session = make_session(limited, reply("Funcionou."), model_retries=1)

    assert session.send(THREAD, "oi").error is None
    assert session.history(THREAD)[-1].text == "Funcionou."


def test_llm_failure_before_any_write_never_changes_data(make_session, service):
    session = make_session(RuntimeError("falha"))
    session.send(THREAD, "Gastei 85")
    assert transaction_count(service) == 0


def test_tool_loop_is_stopped_by_the_call_limit(make_session):
    steps = [call("get_financial_summary", start_month="2026-10") for _ in range(3)]
    session = make_session(*steps, reply("não deve chegar aqui"), model_calls_per_run=3)

    assert session.send(THREAD, "saldo").error is None

    assert len(session.model.calls) == 3
    assert session.pending_approval(THREAD) is None
    assert session.history(THREAD)[-1].text == CALL_LIMIT_MESSAGE


# --- Memória --------------------------------------------------------------------------------


def test_followup_receives_previous_turns_from_checkpoint(make_session):
    session = make_session(reply("Você gastou R$ 750."), reply("Em outubro..."))
    session.send(THREAD, "Quanto gastei com alimentação em setembro?")
    session.send(THREAD, "E em outubro?")

    texts = [m.text for m in session.model.calls[1] if isinstance(m, (HumanMessage, AIMessage))]
    assert texts == [
        "Quanto gastei com alimentação em setembro?",
        "Você gastou R$ 750.",
        "E em outubro?",
    ]


def test_threads_are_isolated(make_session):
    session = make_session(reply("A"), reply("B"))
    session.send("a", "oi de a")
    session.send("b", "oi de b")
    assert "oi de a" not in [m.text for m in session.model.calls[1]]


def test_conversation_and_pending_approval_survive_restart(service, tmp_path):
    """Equivale a reiniciar o processo: novo grafo, nova conexão, mesmo arquivo de checkpoint."""
    path = tmp_path / "state.db"
    first_saver = open_checkpointer(path)
    first = AgentSession(build_agent(scripted(call("create_transaction", **PIZZA)), first_saver), service)
    first.send(THREAD, "Gastei 85")
    interrupt_id = first.pending_approval(THREAD).interrupt_id
    first_saver.conn.close()

    second_saver = open_checkpointer(path)
    second = AgentSession(build_agent(scripted(reply("Registrado.")), second_saver), service)

    assert second.history(THREAD)[0].text == "Gastei 85"
    assert second.pending_approval(THREAD).interrupt_id == interrupt_id
    second.decide(THREAD, interrupt_id, approve=True)
    assert transaction_count(service) == 1
    second_saver.conn.close()


def test_financial_and_agent_data_live_in_separate_files(service, checkpointer):
    finance_tables = {
        row[0]
        for row in sqlite3.connect(service._db.path).execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "checkpoints" not in finance_tables
    assert "transactions" in finance_tables


# --- Structured output ----------------------------------------------------------------------


def test_extraction_returns_validated_draft_and_does_not_write(service):
    model = scripted(
        call(
            "TransactionDraft",
            kind="expense",
            amount="85.50",
            description="Pizza",
            category="Alimentação",
            occurred_on="2026-10-14",
        )
    )

    draft = extract_transaction_draft(model, service, "Pizza 85,50 ontem")

    assert isinstance(draft, TransactionDraft)
    assert str(draft.amount) == "85.50"
    assert draft.occurred_on == date(2026, 10, 14)
    assert transaction_count(service) == 0
    assert "2026-10-15" in model.calls[0][0].text
