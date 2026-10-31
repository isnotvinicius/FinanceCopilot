"""Regras financeiras. Usado igualmente pelas tools do agente e pelos formulários da interface.

Todo cálculo fica aqui, em Python sobre centavos inteiros, para que nenhum número apresentado
ao usuário dependa da LLM.
"""

import calendar
import sqlite3
import unicodedata
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import TypeVar

from pydantic import BaseModel

from finance_copilot.config import today_local
from finance_copilot.database import Database
from finance_copilot.errors import InvalidOperationError, NotFoundError
from finance_copilot.money import MAX_AMOUNT_CENTS, apply_percent, percent_of
from finance_copilot.repository import Repository
from finance_copilot.schemas import (
    AdjustmentOutcome,
    AmountDelta,
    BudgetAdjustment,
    BudgetSimulation,
    Category,
    CategoryDelta,
    CategorySummary,
    CategoryTotal,
    FinancialSummary,
    MoneyView,
    PeriodComparison,
    SalaryResult,
    TransactionChanges,
    TransactionFilter,
    TransactionInput,
    TransactionKind,
    TransactionPage,
    TransactionSource,
    TransactionView,
    TransactionWriteResult,
    money_or_none,
)

MAX_MONTHS_IN_RANGE = 36
MAX_SIMULATION_MONTHS = 120

ResultT = TypeVar("ResultT", bound=BaseModel)


def format_year_month(value: date) -> str:
    return value.strftime("%Y-%m")


def month_bounds(year_month: str) -> tuple[date, date]:
    year, month = int(year_month[:4]), int(year_month[5:7])
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def shift_month(year_month: str, delta: int) -> str:
    index = int(year_month[:4]) * 12 + int(year_month[5:7]) - 1 + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def months_between(start_month: str, end_month: str) -> list[str]:
    if start_month > end_month:
        raise InvalidOperationError(f"Período inválido: {start_month} é posterior a {end_month}.")
    months = [start_month]
    while months[-1] < end_month:
        months.append(shift_month(months[-1], 1))
        if len(months) > MAX_MONTHS_IN_RANGE:
            raise InvalidOperationError(f"O período é limitado a {MAX_MONTHS_IN_RANGE} meses.")
    return months


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _to_view(row: sqlite3.Row) -> TransactionView:
    return TransactionView(
        id=row["id"],
        kind=row["kind"],
        amount=MoneyView.of(row["amount_cents"]),
        description=row["description"],
        category=row["category"],
        occurred_on=row["occurred_on"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class FinanceService:
    def __init__(self, db: Database, clock: Callable[[], date] = today_local) -> None:
        self._db = db
        self._clock = clock

    def today(self) -> date:
        return self._clock()

    def current_month(self) -> str:
        return format_year_month(self.today())

    # --- Categorias ---------------------------------------------------------------------

    def list_categories(self, kind: TransactionKind | None = None) -> list[Category]:
        with self._db.read() as conn:
            rows = Repository(conn).list_categories()
        categories = [Category(**dict(row)) for row in rows]
        return [c for c in categories if kind is None or c.kind == kind]

    def _resolve_category(self, name: str, kind: TransactionKind | None) -> Category:
        """Aceita diferenças de caixa e acento ('alimentacao'); o erro lista as opções válidas."""
        wanted = _normalize(name)
        for category in self.list_categories(kind):
            if _normalize(category.name) == wanted:
                return category
        options = ", ".join(c.name for c in self.list_categories(kind))
        raise InvalidOperationError(f"Categoria '{name}' não existe. Opções válidas: {options}.")

    # --- Escritas -----------------------------------------------------------------------

    def _apply_once(
        self,
        idempotency_key: str,
        operation: str,
        result_type: type[ResultT],
        apply: Callable[[Repository], ResultT],
    ) -> ResultT:
        """Executa `apply` no máximo uma vez por chave.

        O registro da operação e a alteração dos dados são gravados na mesma transação: ou os
        dois persistem ou nenhum. Uma repetição devolve o resultado original com replayed=True.
        """
        with self._db.transaction() as conn:
            repo = Repository(conn)
            stored = repo.get_applied_operation(idempotency_key)
            if stored is not None:
                return result_type.model_validate_json(stored).model_copy(
                    update={"replayed": True}
                )
            result = apply(repo)
            repo.save_applied_operation(idempotency_key, operation, result.model_dump_json())
            return result

    def _check_date(self, occurred_on: date) -> None:
        if occurred_on > self.today():
            raise InvalidOperationError(
                f"A data {occurred_on:%d/%m/%Y} está no futuro. Registre apenas movimentações "
                "que já aconteceram."
            )

    def validate_transaction(self, data: TransactionInput) -> Category:
        """Regras que dependem de estado; também usada para alertar antes da confirmação."""
        self._check_date(data.occurred_on)
        return self._resolve_category(data.category, data.kind)

    def create_transaction(
        self, data: TransactionInput, *, idempotency_key: str, source: TransactionSource
    ) -> TransactionWriteResult:
        category = self.validate_transaction(data)

        def apply(repo: Repository) -> TransactionWriteResult:
            transaction_id = repo.insert_transaction(
                data.kind,
                data.amount_cents,
                data.description,
                category.id,
                data.occurred_on,
                source,
            )
            view = _to_view(repo.get_transaction(transaction_id))
            return TransactionWriteResult(operation="created", transaction=view)

        return self._apply_once(idempotency_key, "create", TransactionWriteResult, apply)

    def update_transaction(
        self, transaction_id: int, changes: TransactionChanges, *, idempotency_key: str
    ) -> TransactionWriteResult:
        if changes.is_empty():
            raise InvalidOperationError("Informe ao menos um campo para atualizar.")
        if changes.occurred_on:
            self._check_date(changes.occurred_on)

        def apply(repo: Repository) -> TransactionWriteResult:
            current = repo.get_transaction(transaction_id)
            if current is None:
                raise NotFoundError(f"Transação #{transaction_id} não encontrada.")
            kind = changes.kind or current["kind"]
            category_name = changes.category or current["category"]
            category = self._resolve_category(category_name, kind)
            repo.update_transaction(
                transaction_id,
                kind,
                changes.amount_cents or current["amount_cents"],
                changes.description or current["description"],
                category.id,
                changes.occurred_on or date.fromisoformat(current["occurred_on"]),
            )
            view = _to_view(repo.get_transaction(transaction_id))
            return TransactionWriteResult(operation="updated", transaction=view)

        return self._apply_once(idempotency_key, "update", TransactionWriteResult, apply)

    def delete_transaction(
        self, transaction_id: int, *, idempotency_key: str
    ) -> TransactionWriteResult:
        """Exclusão lógica: some de todas as consultas, mas a linha permanece para auditoria."""

        def apply(repo: Repository) -> TransactionWriteResult:
            current = repo.get_transaction(transaction_id)
            if current is None:
                raise NotFoundError(f"Transação #{transaction_id} não encontrada.")
            repo.soft_delete_transaction(transaction_id)
            return TransactionWriteResult(operation="deleted", transaction=_to_view(current))

        return self._apply_once(idempotency_key, "delete", TransactionWriteResult, apply)

    def set_monthly_salary(
        self, amount_cents: int, *, effective_from: str | None = None, idempotency_key: str
    ) -> SalaryResult:
        if not 0 < amount_cents <= MAX_AMOUNT_CENTS:
            raise InvalidOperationError("O salário deve ser maior que zero e dentro do limite.")
        month = effective_from or self.current_month()

        def apply(repo: Repository) -> SalaryResult:
            previous = repo.salary_for_month(month)
            repo.upsert_salary(month, amount_cents)
            return SalaryResult(
                amount=MoneyView.of(amount_cents),
                effective_from=month,
                previous_amount=money_or_none(previous),
            )

        return self._apply_once(idempotency_key, "set_salary", SalaryResult, apply)

    # --- Consultas ----------------------------------------------------------------------

    def get_salary(self, year_month: str) -> MoneyView | None:
        with self._db.read() as conn:
            return money_or_none(Repository(conn).salary_for_month(year_month))

    def get_transaction(self, transaction_id: int) -> TransactionView:
        with self._db.read() as conn:
            row = Repository(conn).get_transaction(transaction_id)
        if row is None:
            raise NotFoundError(f"Transação #{transaction_id} não encontrada.")
        return _to_view(row)

    def find_identical_transactions(self, data: TransactionInput) -> list[TransactionView]:
        with self._db.read() as conn:
            rows = Repository(conn).find_identical(
                data.kind, data.amount_cents, data.occurred_on, data.description
            )
        return [_to_view(row) for row in rows]

    def list_transactions(self, filters: TransactionFilter) -> TransactionPage:
        category_id = None
        if filters.category:
            category_id = self._resolve_category(filters.category, filters.kind).id
        with self._db.read() as conn:
            repo = Repository(conn)
            rows = repo.list_transactions(filters, category_id)
            totals = {row["kind"]: row for row in repo.transaction_totals(filters, category_id)}
        count = sum(row["count"] for row in totals.values())
        expenses = totals["expense"]["total"] if "expense" in totals else 0
        income = totals["income"]["total"] if "income" in totals else 0
        return TransactionPage(
            items=[_to_view(row) for row in rows],
            total_count=count,
            total_expenses=MoneyView.of(expenses),
            total_income=MoneyView.of(income),
            truncated=count > len(rows),
            data_status="ok" if count else "no_records",
        )

    def financial_summary(self, start_month: str, end_month: str | None = None) -> FinancialSummary:
        """Receitas, despesas e saldo de meses inteiros.

        O salário vem da configuração e é somado uma vez por mês; `transactions` guarda apenas
        receitas adicionais, de modo que ele nunca é contado em duplicidade.
        """
        end_month = end_month or start_month
        months = months_between(start_month, end_month)
        start, end = month_bounds(start_month)[0], month_bounds(end_month)[1]
        with self._db.read() as conn:
            repo = Repository(conn)
            totals = repo.totals_by_kind(start, end)
            salaries = {month: repo.salary_for_month(month) for month in months}

        expense_cents, expense_count = totals.get("expense", (0, 0))
        extra_cents, income_count = totals.get("income", (0, 0))
        missing = [month for month, salary in salaries.items() if salary is None]
        salary_cents = sum(salary for salary in salaries.values() if salary is not None)
        has_salary = len(missing) < len(months)

        if not has_salary and expense_count == 0 and income_count == 0:
            return FinancialSummary(
                start_month=start_month,
                end_month=end_month,
                data_status="no_data",
                salary_total=None,
                additional_income=None,
                total_income=None,
                total_expenses=None,
                balance=None,
                expense_count=0,
                income_count=0,
                months_without_salary=missing,
                notes=["Não há salário configurado nem movimentações neste período."],
            )

        notes = []
        if missing:
            notes.append(
                "Salário não configurado para: "
                f"{', '.join(missing)}. A receita e o saldo estão incompletos."
            )
        if expense_count == 0:
            notes.append("Nenhuma despesa registrada neste período.")
        income_cents = salary_cents + extra_cents
        return FinancialSummary(
            start_month=start_month,
            end_month=end_month,
            data_status="partial" if missing else "ok",
            salary_total=MoneyView.of(salary_cents) if has_salary else None,
            additional_income=MoneyView.of(extra_cents),
            total_income=MoneyView.of(income_cents),
            total_expenses=MoneyView.of(expense_cents),
            balance=MoneyView.of(income_cents - expense_cents),
            expense_count=expense_count,
            income_count=income_count,
            months_without_salary=missing,
            notes=notes,
        )

    def monthly_overview(self, end_month: str, months: int = 6) -> list[FinancialSummary]:
        """Resumo mês a mês, em ordem cronológica, para os gráficos de evolução."""
        return [
            self.financial_summary(shift_month(end_month, offset))
            for offset in range(-(months - 1), 1)
        ]

    def category_summary(
        self,
        start_month: str,
        end_month: str | None = None,
        kind: TransactionKind = "expense",
    ) -> CategorySummary:
        end_month = end_month or start_month
        months_between(start_month, end_month)
        rows = self._category_rows(start_month, end_month, kind)
        grand_total = sum(row["total"] for row in rows)
        items = [
            CategoryTotal(
                category=row["category"],
                total=MoneyView.of(row["total"]),
                count=row["count"],
                share_percent=percent_of(row["total"], grand_total) or Decimal(0),
            )
            for row in rows
        ]
        return CategorySummary(
            start_month=start_month,
            end_month=end_month,
            kind=kind,
            data_status="ok" if rows else "no_records",
            items=items,
            grand_total=MoneyView.of(grand_total) if rows else None,
            top_category=items[0].category if items else None,
        )

    def _category_rows(self, start_month: str, end_month: str, kind: str) -> list[sqlite3.Row]:
        start, end = month_bounds(start_month)[0], month_bounds(end_month)[1]
        with self._db.read() as conn:
            return Repository(conn).totals_by_category(start, end, kind)

    # --- Análises -----------------------------------------------------------------------

    def compare_periods(
        self,
        baseline_start: str,
        target_start: str,
        *,
        baseline_end: str | None = None,
        target_end: str | None = None,
        category: str | None = None,
    ) -> PeriodComparison:
        """Compara dois períodos: target em relação a baseline, no total e por categoria."""
        baseline_end = baseline_end or baseline_start
        target_end = target_end or target_start
        baseline = self.financial_summary(baseline_start, baseline_end)
        target = self.financial_summary(target_start, target_end)
        comparable = baseline.expense_count > 0 and target.expense_count > 0

        expense_rows = {
            "baseline": self._category_rows(baseline_start, baseline_end, "expense"),
            "target": self._category_rows(target_start, target_end, "expense"),
        }
        category_name = self._resolve_category(category, "expense").name if category else None
        by_category = self._category_deltas(expense_rows, comparable, category_name)

        notes = []
        if not comparable:
            notes.append(
                "Comparação de despesas indisponível: ao menos um dos períodos não tem despesas "
                "registradas. Isso é ausência de dados, não necessariamente gasto zero."
            )
        for summary in (baseline, target):
            notes.extend(summary.notes)
        return PeriodComparison(
            baseline=baseline,
            target=target,
            comparable=comparable,
            income=_delta(_cents(baseline.total_income), _cents(target.total_income)),
            expenses=(
                _delta(_cents(baseline.total_expenses), _cents(target.total_expenses))
                if comparable
                else _delta(None, None)
            ),
            balance=_delta(_cents(baseline.balance), _cents(target.balance)),
            by_category=by_category,
            notes=notes,
        )

    @staticmethod
    def _category_deltas(
        rows: dict[str, list[sqlite3.Row]], comparable: bool, only_category: str | None
    ) -> list[CategoryDelta]:
        baseline = {row["category"]: row["total"] for row in rows["baseline"]}
        target = {row["category"]: row["total"] for row in rows["target"]}
        names = sorted(baseline.keys() | target.keys())
        increases = sum(max(target.get(n, 0) - baseline.get(n, 0), 0) for n in names)

        deltas = []
        for name in names:
            if only_category and name != only_category:
                continue
            # Dentro de um período com dados, categoria sem lançamentos é gasto zero de fato.
            before, after = baseline.get(name, 0), target.get(name, 0)
            delta = _delta(before, after) if comparable else _delta(None, None)
            share = None
            if comparable and after > before:
                share = percent_of(after - before, increases)
            deltas.append(
                CategoryDelta(category=name, share_of_increase_percent=share, **dict(delta))
            )
        return sorted(deltas, key=lambda d: d.difference.cents if d.difference else 0, reverse=True)

    def simulate_budget(
        self,
        *,
        base_month: str | None = None,
        adjustments: list[BudgetAdjustment] | None = None,
        extra_monthly_saving_cents: int | None = None,
        months: int = 1,
    ) -> BudgetSimulation:
        """Projeta o efeito de ajustes no orçamento. Não grava nada."""
        if not 1 <= months <= MAX_SIMULATION_MONTHS:
            raise InvalidOperationError(
                f"O horizonte da simulação deve ser de 1 a {MAX_SIMULATION_MONTHS} meses."
            )
        if not adjustments and not extra_monthly_saving_cents:
            raise InvalidOperationError(
                "Informe ao menos um ajuste por categoria ou uma economia mensal."
            )
        base_month = base_month or self.current_month()
        rows = {r["category"]: r["total"] for r in self._category_rows(base_month, base_month, "expense")}

        outcomes = [self._simulate_adjustment(a, rows) for a in adjustments or []]
        notes = [
            f"Sem despesas de '{o.category or 'todas as categorias'}' em {base_month}: "
            "ajuste não calculado."
            for o in outcomes
            if o.data_status == "no_records"
        ]
        monthly = (extra_monthly_saving_cents or 0) + sum(
            o.monthly_saving.cents for o in outcomes if o.monthly_saving
        )
        has_result = bool(extra_monthly_saving_cents) or any(o.monthly_saving for o in outcomes)

        balance = self.financial_summary(base_month).balance
        return BudgetSimulation(
            base_month=base_month,
            months=months,
            adjustments=outcomes,
            extra_monthly_saving=money_or_none(extra_monthly_saving_cents),
            total_monthly_saving=MoneyView.of(monthly) if has_result else None,
            accumulated_saving=MoneyView.of(monthly * months) if has_result else None,
            current_balance=balance,
            simulated_balance=(
                MoneyView.of(balance.cents + monthly) if balance and has_result else None
            ),
            notes=notes,
        )

    def _simulate_adjustment(
        self, adjustment: BudgetAdjustment, rows: dict[str, int]
    ) -> AdjustmentOutcome:
        if adjustment.category:
            name = self._resolve_category(adjustment.category, "expense").name
            current = rows.get(name)
        else:
            name = None
            current = sum(rows.values()) if rows else None

        if current is None:
            return AdjustmentOutcome(
                category=name,
                percent_change=adjustment.percent_change,
                data_status="no_records",
                current_monthly=None,
                simulated_monthly=None,
                monthly_saving=None,
            )
        change = apply_percent(current, adjustment.percent_change)
        return AdjustmentOutcome(
            category=name,
            percent_change=adjustment.percent_change,
            data_status="ok",
            current_monthly=MoneyView.of(current),
            simulated_monthly=MoneyView.of(current + change),
            monthly_saving=MoneyView.of(-change),
        )


def _cents(money: MoneyView | None) -> int | None:
    return money.cents if money else None


def _delta(baseline: int | None, target: int | None) -> AmountDelta:
    if baseline is None or target is None:
        return AmountDelta(
            baseline=money_or_none(baseline),
            target=money_or_none(target),
            difference=None,
            percent_change=None,
        )
    difference = target - baseline
    return AmountDelta(
        baseline=MoneyView.of(baseline),
        target=MoneyView.of(target),
        difference=MoneyView.of(difference),
        percent_change=percent_of(difference, baseline),
    )
