"""Argumentos das tools. O modelo recebe estes schemas e o Pydantic valida o que ele enviar.

As descrições ficam em português porque são instruções lidas pela LLM.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema

from finance_copilot.money import MAX_AMOUNT_CENTS, parse_money
from finance_copilot.schemas import (
    BudgetAdjustment,
    Description,
    TransactionKind,
    YearMonth,
)

# Para o modelo, um número simples; internamente, Decimal validado e nunca float.
Money = Annotated[
    Decimal,
    BeforeValidator(parse_money),
    Field(gt=0, le=Decimal(MAX_AMOUNT_CENTS) / 100),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]


class CreateTransactionArgs(BaseModel):
    kind: TransactionKind = Field(description="'expense' para despesa, 'income' para receita adicional.")
    amount: Money = Field(description="Valor positivo em reais, por exemplo 85.50.")
    description: Description = Field(description="Descrição curta, por exemplo 'Pizza'.")
    category: str = Field(description="Nome de uma categoria existente compatível com o tipo.")
    occurred_on: date = Field(description="Data da movimentação, YYYY-MM-DD. Não pode ser futura.")


class UpdateTransactionArgs(BaseModel):
    transaction_id: int = Field(gt=0, description="Id obtido em get_transactions.")
    kind: TransactionKind | None = Field(default=None, description="Novo tipo, se mudar.")
    amount: Money | None = Field(default=None, description="Novo valor em reais, se mudar.")
    description: Description | None = Field(default=None, description="Nova descrição, se mudar.")
    category: str | None = Field(default=None, description="Nova categoria, se mudar.")
    occurred_on: date | None = Field(default=None, description="Nova data YYYY-MM-DD, se mudar.")


class DeleteTransactionArgs(BaseModel):
    transaction_id: int = Field(gt=0, description="Id obtido em get_transactions.")


class SetMonthlySalaryArgs(BaseModel):
    amount: Money = Field(description="Salário mensal em reais.")
    effective_from: YearMonth | None = Field(
        default=None,
        description="Mês YYYY-MM a partir do qual vale. Omita para o mês atual.",
    )


class GetTransactionsArgs(BaseModel):
    start_date: date | None = Field(default=None, description="Data inicial YYYY-MM-DD, inclusiva.")
    end_date: date | None = Field(default=None, description="Data final YYYY-MM-DD, inclusiva.")
    category: str | None = Field(default=None, description="Filtrar por categoria.")
    kind: TransactionKind | None = Field(default=None, description="Filtrar por tipo.")
    search: str | None = Field(default=None, description="Trecho da descrição.")
    limit: int = Field(default=20, ge=1, le=100, description="Máximo de itens retornados.")


class PeriodArgs(BaseModel):
    start_month: YearMonth = Field(description="Primeiro mês, YYYY-MM.")
    end_month: YearMonth | None = Field(
        default=None, description="Último mês, YYYY-MM. Omita para um único mês."
    )


class CategorySummaryArgs(PeriodArgs):
    kind: TransactionKind = Field(default="expense", description="Tipo agrupado.")


class ComparePeriodsArgs(BaseModel):
    baseline_start: YearMonth = Field(description="Primeiro mês do período de referência (o mais antigo).")
    baseline_end: YearMonth | None = Field(default=None, description="Último mês do período de referência.")
    target_start: YearMonth = Field(description="Primeiro mês do período analisado.")
    target_end: YearMonth | None = Field(default=None, description="Último mês do período analisado.")
    category: str | None = Field(default=None, description="Limita o detalhamento a uma categoria.")


class SimulateBudgetArgs(BaseModel):
    base_month: YearMonth | None = Field(
        default=None, description="Mês usado como base. Omita para o mês atual."
    )
    adjustments: list[BudgetAdjustment] | None = Field(
        default=None, description="Variações percentuais por categoria."
    )
    extra_monthly_saving: Money | None = Field(
        default=None, description="Economia mensal fixa, em reais, a acumular."
    )
    months: int = Field(default=1, ge=1, le=120, description="Horizonte da projeção em meses.")


class TransactionDraft(BaseModel):
    """Rascunho extraído de texto livre, para pré-preencher o formulário manual."""

    kind: TransactionKind = Field(description="'expense' ou 'income'.")
    amount: Decimal | None = Field(default=None, gt=0, description="Valor em reais, se informado.")
    description: str | None = Field(default=None, description="Descrição curta.")
    category: str | None = Field(default=None, description="Categoria mais adequada da lista.")
    occurred_on: date | None = Field(default=None, description="Data YYYY-MM-DD, se informada.")
