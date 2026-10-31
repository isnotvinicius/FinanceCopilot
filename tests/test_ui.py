"""Testes de ponta a ponta da interface com o AppTest do Streamlit (sem navegador).

A página real roda sobre grafo, checkpoint e banco reais; só a LLM é roteirizada.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from finance_copilot.agent.builder import build_agent, open_checkpointer
from finance_copilot.agent.session import AgentSession, TurnResult
from finance_copilot.bootstrap import App
from finance_copilot.config import Settings
from finance_copilot.database import Database
from finance_copilot.schemas import TransactionFilter
from finance_copilot.seed import seed
from finance_copilot.service import FinanceService
from tests.fakes import call, reply, scripted

APP_FILE = str(Path(__file__).parent.parent / "app.py")


class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.service = FinanceService(Database(tmp_path / "finance.db"))
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

    def start(self, *steps, with_model=True) -> AppTest:
        model = scripted(*steps) if with_model else None
        session = None
        if with_model:
            graph = build_agent(model, open_checkpointer(self.tmp_path / "state.db"), model_retries=0)
            session = AgentSession(graph, self.service)
        app = App(Settings(_env_file=None), self.service, model, session, False)
        self.monkeypatch.setattr("finance_copilot.ui.app.build_app", lambda: app)
        st.cache_resource.clear()
        at = AppTest.from_file(APP_FILE, default_timeout=30)
        at.run()
        assert not at.exception
        return at

    def count(self) -> int:
        return self.service.list_transactions(TransactionFilter(limit=500)).total_count

    def expense_args(self) -> dict:
        return {
            "kind": "expense",
            "amount": 85,
            "description": "Pizza",
            "category": "Alimentação",
            "occurred_on": self.service.today().isoformat(),
        }


@pytest.fixture
def harness(tmp_path, monkeypatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_dashboard_renders_real_values_without_llm(harness):
    seed(harness.service)
    at = harness.start(with_model=False)

    summary = harness.service.financial_summary(harness.service.current_month())
    assert metric(at, "Receita total") == summary.total_income.brl
    assert metric(at, "Despesas") == summary.total_expenses.brl
    assert metric(at, "Saldo disponível") == summary.balance.brl
    assert any("chave de API" in info.value for info in at.info)
    assert len(at.dataframe) >= 2


def test_money_in_markdown_is_escaped_and_delta_keeps_text(harness):
    """Dois "R$" sem escape no mesmo texto viram fórmula LaTeX e corrompem os valores."""
    seed(harness.service)
    at = harness.start(with_model=False)

    caption = next(c.value for c in at.caption if "movimentação(ões)" in c.value)
    assert "Despesas: R\\$ " in caption
    assert "Receitas adicionais: R\\$ " in caption
    assert "vs. mês anterior" in next(m for m in at.metric if m.label == "Despesas").delta


def test_empty_database_shows_no_data_instead_of_zero(harness):
    at = harness.start(with_model=False)
    assert metric(at, "Saldo disponível") == "—"


def test_minimum_flow_chat_confirm_save_dashboard_and_query(harness):
    """O fluxo mínimo do MVP, de ponta a ponta."""
    at = harness.start(
        lambda messages: call("create_transaction", **harness.expense_args()),
        reply("Despesa registrada."),
        call("get_transactions", search="Pizza"),
        reply("Você gastou R$ 85,00 com pizza."),
    )
    assert metric(at, "Saldo disponível") == "—"

    # 1. Pede o cadastro pelo chat: nada é gravado, e o cartão aparece.
    at.chat_input[0].set_value("Gastei 85 reais com pizza hoje").run()
    assert not at.exception
    assert harness.count() == 0
    assert "Confirme a operação" in markdown_text(at)
    assert "R\\$ 85,00" in markdown_text(at)
    assert at.chat_input[0].disabled is True

    # 2. Recarregar a página durante a pausa não grava nem perde o cartão.
    at.run()
    assert harness.count() == 0
    assert "Confirme a operação" in markdown_text(at)

    # 3. A aprovação grava e atualiza o dashboard na mesma tela.
    button(at, "Aprovar").click().run()
    assert not at.exception
    assert harness.count() == 1
    assert metric(at, "Despesas") == "R$ 85,00"
    assert "Despesa registrada." in markdown_text(at)
    assert at.chat_input[0].disabled is False

    # 4. Novas reexecuções não duplicam.
    at.run()
    at.run()
    assert harness.count() == 1

    # 5. A consulta em linguagem natural usa o dado real.
    at.chat_input[0].set_value("Quanto gastei com pizza?").run()
    assert "Você gastou R\\$ 85,00 com pizza." in markdown_text(at)
    assert not at.exception


def test_reject_in_interface_does_not_save(harness):
    at = harness.start(
        lambda messages: call("create_transaction", **harness.expense_args()), reply("Ok, cancelado.")
    )
    at.chat_input[0].set_value("Gastei 85").run()

    button(at, "Rejeitar").click().run()

    assert harness.count() == 0
    assert "Ok, cancelado." in markdown_text(at)
    assert at.chat_input[0].disabled is False


def test_tool_events_are_visible_in_chat(harness):
    at = harness.start(
        call("get_financial_summary", start_month=harness.service.current_month()), reply("Sem dados.")
    )
    at.chat_input[0].set_value("Qual meu saldo?").run()
    assert any("Ferramentas usadas (1)" in e.label for e in at.expander)
    assert "`get_financial_summary`" in markdown_text(at)


def test_simulation_is_flagged_in_interface(harness):
    at = harness.start(
        call("simulate_budget", extra_monthly_saving=300, months=6), reply("Simulação: R$ 1.800,00.")
    )
    at.chat_input[0].set_value("Se eu economizar 300 por mês?").run()
    assert any("SIMULAÇÃO" in w.value for w in at.warning)


def test_llm_failure_shows_message_and_app_survives(harness):
    # Exceção que atravessa o retry: chega à sessão, não à tela como traceback.
    harness.monkeypatch.setattr(
        AgentSession, "_run", lambda self, *args: TurnResult(error="Falha simulada")
    )
    at = harness.start(reply("não usado"))
    at.chat_input[0].set_value("oi").run()
    assert not at.exception
    assert any("Falha simulada" in e.value for e in at.error)


def test_manual_form_saves_once_and_updates_kpis(harness):
    at = harness.start(with_model=False)

    at.text_input(key="new_amount").set_value("120,50")
    at.text_input(key="new_description").set_value("Farmácia")
    at.selectbox(key="new_category_expense").set_value("Saúde")
    next(b for b in at.button if b.label == "Salvar").click().run()

    assert not at.exception
    assert harness.count() == 1
    assert metric(at, "Despesas") == "R$ 120,50"
    assert at.text_input(key="new_description").value == ""
    at.run()
    assert harness.count() == 1


@pytest.mark.parametrize("amount", ["abc", "-5", "0", ""])
def test_manual_form_rejects_invalid_amount(harness, amount):
    at = harness.start(with_model=False)
    at.text_input(key="new_amount").set_value(amount)
    at.text_input(key="new_description").set_value("Teste")
    next(b for b in at.button if b.label == "Salvar").click().run()
    assert not at.exception
    assert harness.count() == 0
    assert at.error
    # O erro não apaga o que o usuário digitou.
    assert at.text_input(key="new_description").value == "Teste"


def test_salary_form_updates_income_without_creating_transaction(harness):
    at = harness.start(with_model=False)
    next(t for t in at.text_input if t.label == "Novo salário (R$)").set_value("5.000,00")
    next(c for c in at.checkbox if "Confirmo o novo salário" in c.label).check()
    next(b for b in at.button if b.label == "Salvar salário").click().run()

    assert not at.exception
    assert metric(at, "Receita total") == "R$ 5.000,00"
    assert harness.count() == 0


def test_delete_requires_confirmation(harness):
    seed(harness.service)
    before = harness.count()
    at = harness.start(with_model=False)

    delete = button(at, "Excluir")
    assert delete.disabled is True

    next(c for c in at.checkbox if "Confirmo a exclusão" in c.label).check().run()
    button(at, "Excluir").click().run()

    assert not at.exception
    assert harness.count() == before - 1


def test_edit_requires_confirmation_then_updates(harness):
    seed(harness.service)
    at = harness.start(with_model=False)
    selected = at.selectbox(key="edit_selected").value
    amount_input = next(t for t in at.text_input if t.label == "Valor (R$)" and t.value)

    amount_input.set_value("1,23")
    next(b for b in at.button if b.label == "Salvar alteração").click().run()
    assert any("Marque a confirmação" in w.value for w in at.warning)
    assert harness.service.get_transaction(selected.id).amount.cents == selected.amount.cents

    next(t for t in at.text_input if t.label == "Valor (R$)" and t.value).set_value("1,23")
    next(c for c in at.checkbox if c.label == "Confirmo a alteração").check()
    next(b for b in at.button if b.label == "Salvar alteração").click().run()

    assert not at.exception
    assert harness.service.get_transaction(selected.id).amount.cents == 123


def test_quick_fill_only_prefills_form(harness):
    at = harness.start(
        call(
            "TransactionDraft",
            kind="expense",
            amount="32.90",
            description="Uber",
            category="Transporte",
        )
    )
    next(t for t in at.text_input if t.label == "Preencher com IA").set_value("Uber 32,90")
    next(b for b in at.button if b.label == "Preencher").click().run()

    assert not at.exception
    assert harness.count() == 0
    assert at.text_input(key="new_amount").value == "32,90"
    assert at.text_input(key="new_description").value == "Uber"
    assert at.selectbox(key="new_category_expense").value == "Transporte"
