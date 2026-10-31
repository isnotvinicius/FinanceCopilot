"""Gráficos do dashboard. Recebem dados já calculados pelo serviço; a LLM não participa.

Os valores em reais aqui são apenas coordenadas de desenho; os textos exibidos vêm de MoneyView.
Mês sem dados vira None (lacuna no gráfico), nunca zero.
"""

import plotly.graph_objects as go

from finance_copilot.schemas import CategorySummary, FinancialSummary, MoneyView

SERIES_INCOME = "#2a78d6"
SERIES_EXPENSE = "#eb6834"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

_MONTH_ABBR = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def month_label(year_month: str) -> str:
    return f"{_MONTH_ABBR[int(year_month[5:7]) - 1]}/{year_month[2:4]}"


def _reais(money: MoneyView | None) -> float | None:
    return None if money is None else money.cents / 100


def _text(money: MoneyView | None) -> str:
    return "Sem dados" if money is None else money.brl


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 16, "t": 32, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_SECONDARY, "size": 13},
        separators=",.",
        barcornerradius=4,
        hoverlabel={"bgcolor": "#fcfcfb", "font": {"color": "#0b0b0b"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def _month_axes(fig: go.Figure) -> None:
    fig.update_xaxes(type="category", showgrid=False, linecolor=AXIS, tickfont={"color": INK_MUTED})
    fig.update_yaxes(
        rangemode="tozero",
        gridcolor=GRID,
        zeroline=False,
        tickprefix="R$ ",
        tickformat=",.0f",
        tickfont={"color": INK_MUTED},
    )


def category_bars(summary: CategorySummary) -> go.Figure:
    """Barras ranqueadas: comparar categorias é mais preciso em barras do que em fatias de pizza."""
    items = list(reversed(summary.items))
    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=[item.category for item in items],
            x=[_reais(item.total) for item in items],
            marker_color=SERIES_EXPENSE,
            width=0.55,
            text=[f"{item.total.brl} ({str(item.share_percent).replace('.', ',')}%)" for item in items],
            textposition="outside",
            cliponaxis=False,
            hovertext=[
                f"{item.category}: {item.total.brl}, {item.count} lançamento(s)" for item in items
            ],
            hoverinfo="text",
        )
    )
    _base_layout(fig, height=max(220, 44 * len(items) + 40))
    fig.update_layout(margin={"l": 8, "r": 130, "t": 8, "b": 8})
    fig.update_xaxes(visible=False, rangemode="tozero")
    fig.update_yaxes(showgrid=False, linecolor=AXIS)
    return fig


def expense_trend(overview: list[FinancialSummary]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=[month_label(s.start_month) for s in overview],
            y=[_reais(s.total_expenses) for s in overview],
            mode="lines+markers",
            line={"color": SERIES_EXPENSE, "width": 2},
            marker={"size": 9, "color": SERIES_EXPENSE, "line": {"color": "#fcfcfb", "width": 2}},
            connectgaps=False,
            hovertext=[_text(s.total_expenses) for s in overview],
            hoverinfo="x+text",
        )
    )
    _base_layout(fig, height=280)
    _month_axes(fig)
    return fig


def income_vs_expense(overview: list[FinancialSummary]) -> go.Figure:
    months = [month_label(s.start_month) for s in overview]
    fig = go.Figure()
    for name, color, values in (
        ("Receitas", SERIES_INCOME, [s.total_income for s in overview]),
        ("Despesas", SERIES_EXPENSE, [s.total_expenses for s in overview]),
    ):
        fig.add_bar(
            name=name,
            x=months,
            y=[_reais(v) for v in values],
            marker_color=color,
            hovertext=[f"{name}: {_text(v)}" for v in values],
            hoverinfo="x+text",
        )
    _base_layout(fig, height=280)
    fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.08)
    _month_axes(fig)
    return fig
