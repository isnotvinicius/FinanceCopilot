"""Schemas Pydantic: entrada dos serviços e resultados estruturados.

Os resultados carregam `MoneyView` (centavos + texto em BRL) para que a LLM repita o valor já
formatado e nunca precise fazer contas. Os schemas de argumentos das tools estão em
`agent/tool_schemas.py`.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from finance_copilot.money import MAX_AMOUNT_CENTS, format_brl

TransactionKind = Literal["expense", "income"]
TransactionSource = Literal["agent", "manual", "seed"]
YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
AmountCents = Annotated[int, Field(gt=0, le=MAX_AMOUNT_CENTS)]

SIMULATION_DISCLAIMER = (
    "SIMULAÇÃO: valores hipotéticos calculados a partir dos dados reais. Nada foi gravado."
)


class MoneyView(BaseModel):
    cents: int
    brl: str

    @classmethod
    def of(cls, cents: int) -> "MoneyView":
        return cls(cents=cents, brl=format_brl(cents))


def money_or_none(cents: int | None) -> MoneyView | None:
    return None if cents is None else MoneyView.of(cents)


# --- Entradas -------------------------------------------------------------------------------


class TransactionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: TransactionKind
    amount_cents: AmountCents
    description: Description
    category: str
    occurred_on: date


class TransactionChanges(BaseModel):
    """Atualização parcial: campos ausentes permanecem como estão."""

    model_config = ConfigDict(frozen=True)

    kind: TransactionKind | None = None
    amount_cents: AmountCents | None = None
    description: Description | None = None
    category: str | None = None
    occurred_on: date | None = None

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)


class TransactionFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_date: date | None = None
    end_date: date | None = None
    category: str | None = None
    kind: TransactionKind | None = None
    search: str | None = None
    limit: int = Field(default=20, ge=1, le=500)


class BudgetAdjustment(BaseModel):
    category: str | None = Field(
        default=None, description="Categoria de despesa; omita para aplicar a todas as despesas."
    )
    percent_change: Decimal = Field(
        ge=-100, le=500, description="Variação em porcentagem: -20 reduz 20%, 10 aumenta 10%."
    )


# --- Resultados -----------------------------------------------------------------------------


class Category(BaseModel):
    id: int
    name: str
    kind: TransactionKind


class TransactionView(BaseModel):
    id: int
    kind: TransactionKind
    amount: MoneyView
    description: str
    category: str
    occurred_on: date
    source: TransactionSource
    created_at: datetime
    updated_at: datetime


class TransactionWriteResult(BaseModel):
    operation: Literal["created", "updated", "deleted"]
    transaction: TransactionView
    replayed: bool = Field(
        default=False, description="True se a operação já havia sido aplicada e não foi repetida."
    )


class SalaryResult(BaseModel):
    amount: MoneyView
    effective_from: YearMonth
    previous_amount: MoneyView | None
    replayed: bool = False


class TransactionPage(BaseModel):
    items: list[TransactionView]
    total_count: int = Field(description="Total que atende ao filtro, mesmo além do limite.")
    total_expenses: MoneyView
    total_income: MoneyView
    truncated: bool
    data_status: Literal["ok", "no_records"]


class FinancialSummary(BaseModel):
    """Resumo de meses inteiros. Campos None indicam dado inexistente, não valor zero."""

    start_month: YearMonth
    end_month: YearMonth
    data_status: Literal["ok", "partial", "no_data"]
    salary_total: MoneyView | None
    additional_income: MoneyView | None
    total_income: MoneyView | None
    total_expenses: MoneyView | None
    balance: MoneyView | None
    expense_count: int
    income_count: int
    months_without_salary: list[YearMonth]
    notes: list[str] = []


class CategoryTotal(BaseModel):
    category: str
    total: MoneyView
    count: int
    share_percent: Decimal


class CategorySummary(BaseModel):
    start_month: YearMonth
    end_month: YearMonth
    kind: TransactionKind
    data_status: Literal["ok", "no_records"]
    items: list[CategoryTotal] = Field(description="Ordenado do maior para o menor total.")
    grand_total: MoneyView | None
    top_category: str | None


class AmountDelta(BaseModel):
    baseline: MoneyView | None
    target: MoneyView | None
    difference: MoneyView | None = Field(description="target - baseline; None sem dados.")
    percent_change: Decimal | None = Field(description="None quando a base é zero ou sem dados.")


class CategoryDelta(AmountDelta):
    category: str
    share_of_increase_percent: Decimal | None = Field(
        description="Participação da categoria na soma dos aumentos; None se não houve aumento."
    )


class PeriodComparison(BaseModel):
    baseline: FinancialSummary
    target: FinancialSummary
    comparable: bool = Field(description="False se um dos períodos não tem despesas registradas.")
    income: AmountDelta
    expenses: AmountDelta
    balance: AmountDelta
    by_category: list[CategoryDelta] = Field(description="Ordenado do maior aumento à maior queda.")
    notes: list[str] = []


class AdjustmentOutcome(BaseModel):
    category: str | None
    percent_change: Decimal
    data_status: Literal["ok", "no_records"]
    current_monthly: MoneyView | None
    simulated_monthly: MoneyView | None
    monthly_saving: MoneyView | None = Field(description="Negativo quando a despesa aumenta.")


class BudgetSimulation(BaseModel):
    is_simulation: Literal[True] = True
    disclaimer: str = SIMULATION_DISCLAIMER
    base_month: YearMonth
    months: int
    adjustments: list[AdjustmentOutcome]
    extra_monthly_saving: MoneyView | None
    total_monthly_saving: MoneyView | None
    accumulated_saving: MoneyView | None
    current_balance: MoneyView | None
    simulated_balance: MoneyView | None
    notes: list[str] = []
