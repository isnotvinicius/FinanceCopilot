"""Indicadores e gráficos. Tudo é lido do SQLite a cada renderização, sem cache: qualquer escrita
seguida de st.rerun() já aparece atualizada."""

import pandas as pd
import streamlit as st

from finance_copilot.config import today_local
from finance_copilot.service import FinanceService, shift_month
from finance_copilot.ui import charts
from finance_copilot.ui.text import escape_markdown

MONTHS = [
    "Janeiro",
    "Fevereiro",
    "Março",
    "Abril",
    "Maio",
    "Junho",
    "Julho",
    "Agosto",
    "Setembro",
    "Outubro",
    "Novembro",
    "Dezembro",
]
_CHART_CONFIG = {"displayModeBar": False, "locale": "pt-BR"}


def render_month_filter() -> str:
    today = today_local()
    years = list(range(today.year, today.year - 5, -1))
    year_col, month_col, _ = st.columns([1, 1, 2])
    year = year_col.selectbox("Ano", years, key="filter_year")
    month = month_col.selectbox(
        "Mês",
        range(1, 13),
        index=today.month - 1,
        format_func=lambda number: MONTHS[number - 1],
        key="filter_month",
    )
    return f"{year:04d}-{month:02d}"


def render_kpis(service: FinanceService, month: str) -> None:
    summary = service.financial_summary(month)
    income_col, expense_col, balance_col = st.columns(3)

    if summary.data_status == "no_data":
        for col, label in (
            (income_col, "Receita total"),
            (expense_col, "Despesas"),
            (balance_col, "Saldo disponível"),
        ):
            col.metric(label, "—", border=True)
        st.info("Não há salário configurado nem movimentações neste mês.")
        return

    salary = summary.salary_total.brl if summary.salary_total else "não configurado"
    income_col.metric(
        "Receita total",
        summary.total_income.brl,
        help=escape_markdown(f"Salário: {salary} · Receitas adicionais: {summary.additional_income.brl}"),
        border=True,
    )
    comparison = service.compare_periods(shift_month(month, -1), month)
    change = comparison.expenses.percent_change
    expense_col.metric(
        "Despesas",
        summary.total_expenses.brl,
        delta=f"{str(change).replace('.', ',')}% vs. mês anterior" if change is not None else None,
        delta_color="inverse",
        border=True,
    )
    balance_col.metric("Saldo disponível", summary.balance.brl, border=True)
    if summary.data_status == "partial":
        st.warning(escape_markdown(" ".join(summary.notes)))


def render_overview(service: FinanceService, month: str) -> None:
    category = service.category_summary(month)
    overview = service.monthly_overview(month, months=6)
    st.markdown("**Gastos por categoria**")
    if category.data_status == "no_records":
        st.caption("Nenhuma despesa registrada neste mês.")
    else:
        st.plotly_chart(
            charts.category_bars(category), width="stretch", config=_CHART_CONFIG, key="chart_categories"
        )

    trend_col, comparison_col = st.columns(2)
    with trend_col:
        st.markdown("**Evolução das despesas**")
        st.plotly_chart(
            charts.expense_trend(overview), width="stretch", config=_CHART_CONFIG, key="chart_trend"
        )
    with comparison_col:
        st.markdown("**Receitas × despesas**")
        st.plotly_chart(
            charts.income_vs_expense(overview), width="stretch", config=_CHART_CONFIG, key="chart_income"
        )

    with st.expander("Ver dados em tabela"):
        st.dataframe(_overview_table(overview), hide_index=True, width="stretch")


def _money_text(money) -> str:
    return money.brl if money else "Sem dados"


def _overview_table(overview) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Mês": charts.month_label(s.start_month),
                "Receitas": _money_text(s.total_income),
                "Despesas": _money_text(s.total_expenses),
                "Saldo": _money_text(s.balance),
            }
            for s in overview
        ]
    )

