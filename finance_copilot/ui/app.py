"""Página principal: dashboard à esquerda e copilot à direita, com atualização visível lado a lado."""

import streamlit as st

from finance_copilot.bootstrap import App, build_app
from finance_copilot.ui import dashboard, forms, model_picker
from finance_copilot.ui.chat import render_chat


@st.cache_resource(show_spinner=False)
def _get_app() -> App:
    # Um único conjunto de serviços por processo; os dados ficam nos arquivos SQLite.
    return build_app()


@st.cache_resource(show_spinner=False)
def _get_app_for(provider: str, model: str) -> App:
    """Modelo escolhido na barra lateral. Reaproveita o serviço: os dados são os mesmos."""
    base = _get_app()
    return build_app(base.settings.with_llm(provider, model), service=base.service)


def main() -> None:
    st.set_page_config(page_title="Finance Copilot", page_icon="💰", layout="wide")
    base = _get_app()
    selection = model_picker.choose_model(base.settings)
    configured = (base.settings.llm_provider, base.settings.llm_model)
    app = base if selection == configured else _get_app_for(*selection)
    model_picker.render_status(app)

    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash)

    dashboard_col, chat_col = st.columns([3, 2], gap="large")
    with dashboard_col:
        st.title("Finance Copilot", anchor=False)
        st.caption("Dados fictícios de demonstração · BRL · America/Sao_Paulo")
        month = dashboard.render_month_filter()
        dashboard.render_kpis(app.service, month)
        overview_tab, history_tab, new_tab, salary_tab = st.tabs(
            ["Visão geral", "Histórico", "Novo lançamento", "Salário"]
        )
        with overview_tab:
            dashboard.render_overview(app.service, month)
        with history_tab:
            forms.render_history(app.service, month)
        with new_tab:
            forms.render_new_transaction(app)
        with salary_tab:
            forms.render_salary(app.service, month)
    with chat_col:
        render_chat(app)
