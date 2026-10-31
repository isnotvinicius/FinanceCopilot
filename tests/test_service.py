from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from finance_copilot.errors import InvalidOperationError, NotFoundError
from finance_copilot.schemas import (
    BudgetAdjustment,
    TransactionChanges,
    TransactionFilter,
    TransactionInput,
)
from finance_copilot.seed import seed

SEP = date(2026, 9, 10)
OCT = date(2026, 10, 10)


def page(service, **filters):
    return service.list_transactions(TransactionFilter(**filters))


# --- Cadastro e consulta --------------------------------------------------------------------


def test_create_and_list_transaction(service, builder):
    created = builder.add(8500, description="Pizza", on=date(2026, 10, 14))

    assert created.id > 0
    assert created.amount.cents == 8500
    assert created.amount.brl == "R$ 85,00"
    assert created.category == "Alimentação"
    result = page(service)
    assert [t.description for t in result.items] == ["Pizza"]
    assert result.total_expenses.cents == 8500
    assert result.data_status == "ok"


def test_filters_by_period_category_kind_and_text(service, builder):
    builder.add(1000, "Alimentação", SEP, description="Mercado")
    builder.add(2000, "Lazer", OCT, description="Cinema")
    builder.add(3000, "Freelance", OCT, kind="income", description="Site")

    assert page(service, start_date=OCT).total_count == 2
    assert page(service, end_date=SEP).total_count == 1
    assert [t.description for t in page(service, category="lazer").items] == ["Cinema"]
    assert page(service, kind="income").total_income.cents == 3000
    assert page(service, search="merc").total_count == 1


def test_search_treats_wildcards_literally(service, builder):
    builder.add(1000, description="Mercado")
    assert page(service, search="%").total_count == 0
    assert page(service, search="_").total_count == 0


def test_totals_consider_rows_beyond_the_limit(service, builder):
    for _ in range(3):
        builder.add(1000)
    result = page(service, limit=2)
    assert len(result.items) == 2
    assert result.truncated is True
    assert result.total_count == 3
    assert result.total_expenses.cents == 3000


def test_sql_injection_attempts_are_just_data(service, builder):
    payload = "x'; DROP TABLE transactions; --"
    builder.add(1000, description=payload)
    assert page(service, search=payload).total_count == 1
    assert page(service).total_count == 1
    with pytest.raises(InvalidOperationError):
        page(service, category=payload)


# --- Validação ------------------------------------------------------------------------------


@pytest.mark.parametrize("cents", [0, -100, 100_000_000_01])
def test_invalid_amounts_are_rejected(cents):
    with pytest.raises(ValidationError):
        TransactionInput(
            kind="expense", amount_cents=cents, description="X", category="Lazer", occurred_on=OCT
        )


def test_blank_description_is_rejected():
    with pytest.raises(ValidationError):
        TransactionInput(
            kind="expense", amount_cents=1, description="   ", category="Lazer", occurred_on=OCT
        )


def test_database_constraint_also_rejects_invalid_amount(service):
    """A regra de valor positivo não depende apenas do Pydantic."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError), service._db.transaction() as conn:
        conn.execute(
            """INSERT INTO transactions (kind, amount_cents, description, category_id, occurred_on,
               source, created_at, updated_at) VALUES ('expense', -1, 'x', 1, '2026-10-01',
               'manual', 'now', 'now')"""
        )


def test_future_date_is_rejected(builder):
    with pytest.raises(InvalidOperationError, match="futuro"):
        builder.add(1000, on=date(2026, 10, 16))


def test_unknown_category_error_lists_options(builder):
    with pytest.raises(InvalidOperationError, match="Opções válidas.*Alimentação"):
        builder.add(1000, category="Pets")


def test_category_must_match_transaction_kind(builder):
    with pytest.raises(InvalidOperationError):
        builder.add(1000, category="Freelance", kind="expense")


def test_category_lookup_ignores_case_and_accents(builder):
    assert builder.add(1000, category="alimentacao").category == "Alimentação"


# --- Edição e exclusão ----------------------------------------------------------------------


def test_update_changes_only_informed_fields(service, builder):
    created = builder.add(1000, "Lazer", description="Cinema")
    updated = service.update_transaction(
        created.id, TransactionChanges(amount_cents=2500), idempotency_key="u1"
    ).transaction
    assert updated.amount.cents == 2500
    assert updated.description == "Cinema"
    assert updated.category == "Lazer"


def test_update_requires_some_change_and_existing_row(service, builder):
    created = builder.add(1000)
    with pytest.raises(InvalidOperationError):
        service.update_transaction(created.id, TransactionChanges(), idempotency_key="u1")
    with pytest.raises(NotFoundError):
        service.update_transaction(
            999, TransactionChanges(amount_cents=1), idempotency_key="u2"
        )


def test_failed_operation_does_not_consume_idempotency_key(service, builder):
    with pytest.raises(NotFoundError):
        service.delete_transaction(1, idempotency_key="d1")
    created = builder.add(1000)
    assert created.id == 1
    assert service.delete_transaction(1, idempotency_key="d1").replayed is False


def test_delete_hides_row_from_every_query(service, builder):
    created = builder.add(1000)
    service.delete_transaction(created.id, idempotency_key="d1")

    assert page(service).total_count == 0
    assert service.financial_summary("2026-10").data_status == "no_data"
    with pytest.raises(NotFoundError):
        service.get_transaction(created.id)
    with pytest.raises(NotFoundError):
        service.delete_transaction(created.id, idempotency_key="d2")


# --- Saldo e salário ------------------------------------------------------------------------


def test_balance_counts_salary_once_plus_additional_income(service, builder):
    service.set_monthly_salary(500000, effective_from="2026-10", idempotency_key="s1")
    builder.add(100000, "Freelance", OCT, kind="income")
    builder.add(30000, "Lazer", OCT)

    summary = service.financial_summary("2026-10")

    assert summary.salary_total.cents == 500000
    assert summary.additional_income.cents == 100000
    assert summary.total_income.cents == 600000
    assert summary.total_expenses.cents == 30000
    assert summary.balance.cents == 570000
    assert summary.data_status == "ok"


@pytest.mark.parametrize("cents", [0, -1, 100_000_000_01])
def test_invalid_salary_is_rejected(service, cents):
    with pytest.raises(InvalidOperationError):
        service.set_monthly_salary(cents, idempotency_key="s1")


def test_salary_is_never_stored_as_transaction(service):
    service.set_monthly_salary(500000, effective_from="2026-10", idempotency_key="s1")
    assert page(service).total_count == 0
    assert service.financial_summary("2026-10").additional_income.cents == 0


def test_salary_change_does_not_rewrite_previous_months(service):
    service.set_monthly_salary(500000, effective_from="2026-09", idempotency_key="s1")
    result = service.set_monthly_salary(600000, effective_from="2026-10", idempotency_key="s2")

    assert result.previous_amount.cents == 500000
    assert service.financial_summary("2026-09").salary_total.cents == 500000
    assert service.financial_summary("2026-10").salary_total.cents == 600000
    assert service.financial_summary("2026-11").salary_total.cents == 600000


def test_salary_is_summed_for_each_month_of_a_range(service):
    service.set_monthly_salary(500000, effective_from="2026-09", idempotency_key="s1")
    assert service.financial_summary("2026-09", "2026-10").salary_total.cents == 1000000


def test_no_data_is_distinct_from_zero(service, builder):
    empty = service.financial_summary("2026-10")
    assert empty.data_status == "no_data"
    assert empty.total_expenses is None
    assert empty.balance is None

    service.set_monthly_salary(500000, effective_from="2026-10", idempotency_key="s1")
    zero_expenses = service.financial_summary("2026-10")
    assert zero_expenses.data_status == "ok"
    assert zero_expenses.total_expenses.cents == 0
    assert zero_expenses.expense_count == 0
    assert "Nenhuma despesa" in zero_expenses.notes[0]


def test_missing_salary_marks_summary_as_partial(service, builder):
    builder.add(1000, on=OCT)
    summary = service.financial_summary("2026-10")
    assert summary.data_status == "partial"
    assert summary.salary_total is None
    assert summary.months_without_salary == ["2026-10"]
    assert "incompletos" in summary.notes[0]


@pytest.mark.parametrize(
    ("start", "end"), [("2026-10", "2026-09"), ("2020-01", "2026-10")]
)
def test_invalid_ranges_are_rejected(service, start, end):
    with pytest.raises(InvalidOperationError):
        service.financial_summary(start, end)


# --- Categorias -----------------------------------------------------------------------------


def test_category_summary_totals_and_shares(service, builder):
    builder.add(7500, "Alimentação", OCT)
    builder.add(2500, "Lazer", OCT)
    builder.add(9999, "Lazer", SEP)
    builder.add(5000, "Freelance", OCT, kind="income")

    summary = service.category_summary("2026-10")

    assert summary.top_category == "Alimentação"
    assert [(i.category, i.total.cents, i.share_percent) for i in summary.items] == [
        ("Alimentação", 7500, Decimal("75.0")),
        ("Lazer", 2500, Decimal("25.0")),
    ]
    assert summary.grand_total.cents == 10000


def test_category_summary_without_records(service):
    summary = service.category_summary("2026-10")
    assert summary.data_status == "no_records"
    assert summary.top_category is None
    assert summary.grand_total is None


# --- Comparação -----------------------------------------------------------------------------


def test_compare_months_totals_and_contribution(service, builder):
    builder.add(10000, "Alimentação", SEP)
    builder.add(5000, "Lazer", SEP)
    builder.add(4000, "Transporte", SEP)
    builder.add(13000, "Alimentação", OCT)
    builder.add(6000, "Lazer", OCT)
    builder.add(1000, "Saúde", OCT)

    result = service.compare_periods("2026-09", "2026-10")

    assert result.comparable is True
    assert result.expenses.baseline.cents == 19000
    assert result.expenses.target.cents == 20000
    assert result.expenses.difference.cents == 1000
    assert result.expenses.percent_change == Decimal("5.3")
    rows = {d.category: d for d in result.by_category}
    assert [d.category for d in result.by_category] == [
        "Alimentação",
        "Lazer",
        "Saúde",
        "Transporte",
    ]
    assert rows["Alimentação"].share_of_increase_percent == Decimal("60.0")
    assert rows["Saúde"].percent_change is None  # base zero não tem variação definida
    assert rows["Transporte"].difference.cents == -4000
    assert rows["Transporte"].share_of_increase_percent is None


def test_compare_can_focus_on_one_category(service, builder):
    builder.add(10000, "Alimentação", SEP)
    builder.add(5000, "Lazer", SEP)
    builder.add(12000, "Alimentação", OCT)

    result = service.compare_periods("2026-09", "2026-10", category="alimentacao")

    assert [d.category for d in result.by_category] == ["Alimentação"]
    assert result.by_category[0].difference.cents == 2000


def test_compare_without_data_does_not_invent_a_difference(service, builder):
    builder.add(10000, "Alimentação", OCT)

    result = service.compare_periods("2026-09", "2026-10")

    assert result.comparable is False
    assert result.expenses.difference is None
    assert result.by_category[0].difference is None
    assert "ausência de dados" in result.notes[0]


# --- Simulações -----------------------------------------------------------------------------


def test_simulate_percent_reduction_of_category(service, builder):
    service.set_monthly_salary(500000, effective_from="2026-10", idempotency_key="s1")
    builder.add(30000, "Lazer", OCT)

    result = service.simulate_budget(
        base_month="2026-10",
        adjustments=[BudgetAdjustment(category="Lazer", percent_change=Decimal(-20))],
    )

    assert result.is_simulation is True
    assert "SIMULAÇÃO" in result.disclaimer
    assert result.adjustments[0].current_monthly.cents == 30000
    assert result.adjustments[0].simulated_monthly.cents == 24000
    assert result.total_monthly_saving.cents == 6000
    assert result.current_balance.cents == 470000
    assert result.simulated_balance.cents == 476000


def test_simulate_accumulated_monthly_saving(service):
    result = service.simulate_budget(extra_monthly_saving_cents=30000, months=6)
    assert result.accumulated_saving.cents == 180000
    assert result.simulated_balance is None  # sem dados reais, não há saldo para projetar


def test_simulate_increase_results_in_negative_saving(service, builder):
    builder.add(10000, "Lazer", OCT)
    result = service.simulate_budget(
        base_month="2026-10", adjustments=[BudgetAdjustment(percent_change=Decimal(10))]
    )
    assert result.total_monthly_saving.cents == -1000


def test_simulate_without_data_returns_no_values(service):
    result = service.simulate_budget(
        base_month="2026-10",
        adjustments=[BudgetAdjustment(category="Lazer", percent_change=Decimal(-20))],
    )
    assert result.adjustments[0].data_status == "no_records"
    assert result.adjustments[0].monthly_saving is None
    assert result.total_monthly_saving is None
    assert result.accumulated_saving is None
    assert "Sem despesas" in result.notes[0]


def test_simulation_does_not_write(service, builder):
    builder.add(10000, "Lazer", OCT)
    service.simulate_budget(
        base_month="2026-10",
        adjustments=[BudgetAdjustment(category="Lazer", percent_change=Decimal(-50))],
    )
    assert page(service).total_expenses.cents == 10000


def test_simulation_input_validation(service):
    with pytest.raises(InvalidOperationError):
        service.simulate_budget()
    with pytest.raises(InvalidOperationError):
        service.simulate_budget(extra_monthly_saving_cents=100, months=0)
    with pytest.raises(ValidationError):
        BudgetAdjustment(percent_change=Decimal(-101))


# --- Idempotência ---------------------------------------------------------------------------


def _pizza():
    return TransactionInput(
        kind="expense", amount_cents=8500, description="Pizza", category="Alimentação", occurred_on=OCT
    )


def test_repeated_key_does_not_create_duplicate(service):
    first = service.create_transaction(_pizza(), idempotency_key="k1", source="agent")
    second = service.create_transaction(_pizza(), idempotency_key="k1", source="agent")

    assert first.replayed is False
    assert second.replayed is True
    assert second.transaction.id == first.transaction.id
    assert page(service).total_count == 1


def test_different_keys_are_different_operations(service):
    service.create_transaction(_pizza(), idempotency_key="k1", source="agent")
    service.create_transaction(_pizza(), idempotency_key="k2", source="agent")
    assert page(service).total_count == 2


def test_repeated_update_delete_and_salary_are_noops(service, builder):
    created = builder.add(1000)
    changes = TransactionChanges(amount_cents=2000)
    service.update_transaction(created.id, changes, idempotency_key="u1")
    service.update_transaction(created.id, TransactionChanges(amount_cents=3000), idempotency_key="u2")
    replay = service.update_transaction(created.id, changes, idempotency_key="u1")
    assert replay.replayed is True
    assert replay.transaction.amount.cents == 2000
    assert service.get_transaction(created.id).amount.cents == 3000

    service.delete_transaction(created.id, idempotency_key="d1")
    assert service.delete_transaction(created.id, idempotency_key="d1").replayed is True

    service.set_monthly_salary(100, effective_from="2026-10", idempotency_key="s1")
    service.set_monthly_salary(200, effective_from="2026-10", idempotency_key="s2")
    assert service.set_monthly_salary(100, effective_from="2026-10", idempotency_key="s1").replayed
    assert service.get_salary("2026-10").cents == 200


def test_rollback_keeps_data_and_ledger_consistent(service):
    """Se a gravação falha no meio, nem a transação nem o registro da operação persistem."""
    original = service._db.transaction

    def apply_failing(repo):
        repo.insert_transaction("expense", 100, "x", 1, OCT, "manual")
        raise RuntimeError("falha simulada")

    with pytest.raises(RuntimeError):
        service._apply_once("k1", "create", None, apply_failing)

    with original() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM applied_operations").fetchone()[0] == 0


def test_find_identical_is_used_to_warn_about_repeats(service):
    service.create_transaction(_pizza(), idempotency_key="k1", source="agent")
    assert len(service.find_identical_transactions(_pizza())) == 1


# --- Seed -----------------------------------------------------------------------------------


def test_seed_is_reproducible(service):
    assert seed(service) == 21
    assert seed(service) == 0
    assert page(service, limit=100).total_count == 21


def test_seed_never_creates_future_dates_and_covers_both_months(service):
    seed(service)
    items = page(service, limit=100).items
    assert max(t.occurred_on for t in items) <= service.today()
    assert {t.occurred_on.month for t in items} == {9, 10}
    assert service.financial_summary("2026-09").balance.cents > 0
    assert service.compare_periods("2026-09", "2026-10").comparable is True
