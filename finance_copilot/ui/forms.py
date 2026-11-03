"""Histórico, formulários manuais e configuração de salário.

Os formulários usam o mesmo FinanceService das tools. A chave de idempotência é gerada por instância
do formulário e só muda depois de um sucesso: reenvios e reexecuções do Streamlit não duplicam.
"""

from datetime import date
from uuid import uuid4

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from finance_copilot.agent.extraction import extract_transaction_draft
from finance_copilot.bootstrap import App
from finance_copilot.errors import FinanceError
from finance_copilot.llm.errors import classify_llm_error
from finance_copilot.money import format_brl, parse_money, to_cents
from finance_copilot.schemas import (
    TransactionChanges,
    TransactionFilter,
    TransactionInput,
    TransactionView,
)
from finance_copilot.service import FinanceService, month_bounds
from finance_copilot.ui.text import escape_markdown

KIND_LABELS = {"expense": "Despesa", "income": "Receita adicional"}
_ALL = "Todas"
_ANY_KIND = "Todos"


def _form_key(name: str) -> str:
    state_key = f"idempotency_{name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = f"manual:{name}:{uuid4().hex}"
    return st.session_state[state_key]


def _finish_form(name: str, message: str, section: str) -> None:
    """Depois de um sucesso: nova chave para o próximo envio e recarga dos indicadores.

    A mensagem aparece na seção onde o usuário salvou (ver _show_success), após o rerun.
    """
    st.session_state.pop(f"idempotency_{name}", None)
    st.session_state["form_success"] = (section, message)
    st.rerun()


def _show_success(section: str) -> None:
    """Aviso verde de confirmação, exibido uma única vez na seção que salvou."""
    saved = st.session_state.get("form_success")
    if saved and saved[0] == section:
        del st.session_state["form_success"]
        st.success(saved[1], icon="✅")


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "Verifique os campos: valor maior que zero e descrição preenchida."
    return str(exc)


def _parse_amount(text: str) -> int:
    try:
        return to_cents(parse_money(text))
    except ValueError as exc:
        raise FinanceError(f"Valor inválido: use o formato 85,50 ({exc}).") from exc


# --- Histórico ------------------------------------------------------------------------------


def render_history(service: FinanceService, month: str) -> None:
    _show_success("history")
    start_default, end_default = month_bounds(month)
    start_col, end_col, category_col, kind_col = st.columns(4)
    # A chave inclui o mês para que o filtro acompanhe o mês selecionado no topo.
    start = start_col.date_input("De", start_default, format="DD/MM/YYYY", key=f"history_start_{month}")
    end = end_col.date_input("Até", end_default, format="DD/MM/YYYY", key=f"history_end_{month}")
    categories = [_ALL, *(c.name for c in service.list_categories())]
    category = category_col.selectbox("Categoria", categories, key="history_category")
    kind_label = kind_col.selectbox("Tipo", [_ANY_KIND, *KIND_LABELS.values()], key="history_kind")
    kind = next((k for k, label in KIND_LABELS.items() if label == kind_label), None)

    if category != _ALL:
        category_kind = next(c.kind for c in service.list_categories() if c.name == category)
        if kind and kind != category_kind:
            st.info("Essa categoria não pertence ao tipo selecionado.")
            return

    page = service.list_transactions(
        TransactionFilter(
            start_date=start,
            end_date=end,
            category=None if category == _ALL else category,
            kind=kind,
            limit=500,
        )
    )
    if page.data_status == "no_records":
        st.caption("Nenhuma movimentação encontrada com esses filtros.")
        return

    st.caption(
        escape_markdown(
            f"{page.total_count} movimentação(ões) · Despesas: {page.total_expenses.brl} · "
            f"Receitas adicionais: {page.total_income.brl}"
        )
    )
    st.dataframe(_transactions_table(page.items), hide_index=True, width="stretch")
    if page.truncated:
        st.caption("Exibindo as 500 mais recentes; os totais consideram todas.")
    _render_edit_section(service, page.items)


def _transactions_table(items: list[TransactionView]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Id": item.id,
                "Data": f"{item.occurred_on:%d/%m/%Y}",
                "Descrição": item.description,
                "Categoria": item.category,
                "Tipo": KIND_LABELS[item.kind],
                "Valor": item.amount.brl,
            }
            for item in items
        ]
    )


def _render_edit_section(service: FinanceService, items: list[TransactionView]) -> None:
    with st.expander("Editar ou excluir uma movimentação"):
        selected = st.selectbox(
            "Movimentação",
            items,
            format_func=lambda t: f"#{t.id} · {t.occurred_on:%d/%m/%Y} · {t.description} · {t.amount.brl}",
            key="edit_selected",
        )
        if selected is None:
            return
        _render_edit_form(service, selected)
        _render_delete(service, selected)


def _render_edit_form(service: FinanceService, item: TransactionView) -> None:
    # updated_at na chave: os campos recarregam quando o registro muda.
    suffix = f"{item.id}_{item.updated_at.timestamp():.0f}"
    categories = [c.name for c in service.list_categories(item.kind)]
    with st.form(f"edit_form_{suffix}"):
        amount = st.text_input("Valor (R$)", item.amount.brl.replace("R$ ", ""))
        description = st.text_input("Descrição", item.description)
        category = st.selectbox("Categoria", categories, index=categories.index(item.category))
        occurred_on = st.date_input("Data", item.occurred_on, format="DD/MM/YYYY")
        confirmed = st.checkbox("Confirmo a alteração")
        submitted = st.form_submit_button("Salvar alteração")
    if not submitted:
        return
    if not confirmed:
        st.warning("Marque a confirmação para salvar.")
        return
    try:
        service.update_transaction(
            item.id,
            TransactionChanges(
                amount_cents=_parse_amount(amount),
                description=description,
                category=category,
                occurred_on=occurred_on,
            ),
            idempotency_key=_form_key(f"edit_{item.id}"),
        )
    except (FinanceError, ValidationError) as exc:
        st.error(_error_text(exc))
        return
    _finish_form(f"edit_{item.id}", f"Movimentação #{item.id} atualizada com sucesso.", "history")


def _render_delete(service: FinanceService, item: TransactionView) -> None:
    confirmed = st.checkbox(f"Confirmo a exclusão de #{item.id}", key=f"delete_confirm_{item.id}")
    if st.button("Excluir", disabled=not confirmed, key=f"delete_{item.id}"):
        try:
            service.delete_transaction(item.id, idempotency_key=_form_key(f"delete_{item.id}"))
        except FinanceError as exc:
            st.error(str(exc))
            return
        _finish_form(f"delete_{item.id}", f"Movimentação #{item.id} excluída com sucesso.", "history")


# --- Novo lançamento ------------------------------------------------------------------------


def render_new_transaction(app: App) -> None:
    service = app.service
    if st.session_state.pop("new_form_saved", False):
        for key in ("new_amount", "new_description", "new_date"):
            st.session_state.pop(key, None)
    _show_success("new")
    _render_quick_fill(app)
    kind = st.radio(
        "Tipo", list(KIND_LABELS), format_func=KIND_LABELS.get, horizontal=True, key="new_kind"
    )
    with st.form("new_transaction"):
        amount_col, date_col = st.columns(2)
        amount = amount_col.text_input("Valor (R$)", placeholder="85,50", key="new_amount")
        if "new_date" not in st.session_state:
            st.session_state["new_date"] = service.today()
        occurred_on = date_col.date_input("Data", format="DD/MM/YYYY", key="new_date")
        description = st.text_input("Descrição", key="new_description")
        categories = [c.name for c in service.list_categories(kind)]
        category = st.selectbox("Categoria", categories, key=f"new_category_{kind}")
        submitted = st.form_submit_button("Salvar", type="primary")
    if not submitted:
        return
    try:
        amount_cents = _parse_amount(amount)
        service.create_transaction(
            TransactionInput(
                kind=kind,
                amount_cents=amount_cents,
                description=description,
                category=category,
                occurred_on=occurred_on,
            ),
            idempotency_key=_form_key("new"),
            source="manual",
        )
    except (FinanceError, ValidationError) as exc:
        st.error(_error_text(exc))
        return
    st.session_state["new_form_saved"] = True
    _finish_form(
        "new",
        f"{KIND_LABELS[kind]} de {format_brl(amount_cents)} ({description.strip()}) registrada com sucesso.",
        "new",
    )


def _render_quick_fill(app: App) -> None:
    """Structured output: a LLM preenche os campos, e nada é salvo sem o clique do usuário."""
    if app.model is None:
        return
    spec = app.model_spec
    if spec is not None and not spec.supports_structured_output:
        st.caption("Preencher com IA indisponível: o modelo ativo não suporta structured output.")
        return
    with st.form("quick_fill", border=False):
        text_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
        text = text_col.text_input("Preencher com IA", placeholder="Uber de 32,90 hoje de manhã")
        requested = button_col.form_submit_button("Preencher", width="stretch")
    if not requested or not text.strip():
        return
    try:
        with st.spinner("Lendo o texto..."):
            method = spec.structured_output_method if spec else None
            draft = extract_transaction_draft(app.model, app.service, text, method=method)
    except Exception as exc:
        detail = classify_llm_error(exc).message
        st.error(f"Não foi possível interpretar o texto agora. Preencha manualmente. {detail or ''}")
        return
    _apply_draft(app.service, draft)


def _apply_draft(service: FinanceService, draft) -> None:
    # Roda antes da criação dos widgets, único momento em que seu estado pode ser definido.
    st.session_state["new_kind"] = draft.kind
    if draft.amount is not None:
        st.session_state["new_amount"] = f"{draft.amount:.2f}".replace(".", ",")
    if draft.description:
        st.session_state["new_description"] = draft.description
    valid = [c.name for c in service.list_categories(draft.kind)]
    if draft.category in valid:
        st.session_state[f"new_category_{draft.kind}"] = draft.category
    if isinstance(draft.occurred_on, date):
        st.session_state["new_date"] = draft.occurred_on
    st.toast("Campos preenchidos. Revise e clique em Salvar.")


# --- Salário --------------------------------------------------------------------------------


def render_salary(service: FinanceService, month: str) -> None:
    _show_success("salary")
    current = service.get_salary(month)
    st.caption(
        "O salário é uma receita recorrente da configuração: entra uma vez em cada mês e não "
        "aparece como transação. A mudança vale do mês indicado em diante."
    )
    st.metric(f"Salário vigente em {month}", current.brl if current else "Não configurado")
    with st.form("salary_form"):
        amount = st.text_input("Novo salário (R$)", placeholder="6.500,00")
        confirmed = st.checkbox(f"Confirmo o novo salário a partir de {month}")
        submitted = st.form_submit_button("Salvar salário")
    if not submitted:
        return
    if not confirmed:
        st.warning("Marque a confirmação para salvar.")
        return
    try:
        cents = _parse_amount(amount)
        service.set_monthly_salary(cents, effective_from=month, idempotency_key=_form_key("salary"))
    except FinanceError as exc:
        st.error(str(exc))
        return
    _finish_form("salary", f"Salário de {format_brl(cents)} salvo com sucesso a partir de {month}.", "salary")
