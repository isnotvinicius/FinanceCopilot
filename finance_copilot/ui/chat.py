"""Coluna de conversa: mensagens, confirmação de operações e eventos das ferramentas."""

from uuid import uuid4

import streamlit as st

from finance_copilot.agent.session import (
    AgentSession,
    ChatEntry,
    PendingApproval,
    PendingApprovalError,
    StaleApprovalError,
    ToolEvent,
)
from finance_copilot.bootstrap import App
from finance_copilot.schemas import SIMULATION_DISCLAIMER
from finance_copilot.ui.text import escape_markdown

_STATUS = {
    "ok": ("✅", "executada"),
    "error": ("❌", "erro"),
    "rejected": ("🚫", "rejeitada pelo usuário"),
    "pending": ("⏳", "aguardando confirmação"),
}
_SUGGESTIONS = (
    "Exemplos: *Gastei 85 reais com pizza ontem* · *Quanto tenho disponível neste mês?* · "
    "*Quais categorias mais contribuíram para o aumento dos meus gastos?*"
)


def _thread_id() -> str:
    """A conversa fica na URL: ao recarregar a página, o checkpoint restaura tudo."""
    thread_id = st.query_params.get("thread")
    if not thread_id:
        thread_id = uuid4().hex
        st.query_params["thread"] = thread_id
    return thread_id


def render_chat(app: App) -> None:
    title_col, button_col = st.columns([3, 2], vertical_alignment="center")
    title_col.subheader("Copilot", anchor=False)
    session = app.session
    if session is None:
        st.info(
            f"O chat precisa de uma chave de API para o provedor '{app.settings.llm_provider}'. "
            "Copie `.env.example` para `.env`, preencha a chave e reinicie. "
            "O dashboard e os lançamentos manuais funcionam normalmente sem ela."
        )
        return

    thread_id = _thread_id()
    if button_col.button("Nova conversa", width="stretch"):
        st.query_params["thread"] = uuid4().hex
        st.rerun()

    pending = session.pending_approval(thread_id)
    history = st.container(height=560, border=True)
    with history:
        entries = session.history(thread_id)
        if not entries:
            st.caption(_SUGGESTIONS)
        for entry in entries:
            _render_entry(entry)
        if pending:
            _render_approval(session, thread_id, pending)
    _show_chat_error()

    prompt = st.chat_input(
        "Confirme ou rejeite a operação acima" if pending else "Pergunte ou registre algo",
        disabled=bool(pending),
    )
    if prompt:
        with history:
            with st.chat_message("user"):
                st.markdown(escape_markdown(prompt))
            with st.chat_message("assistant"), st.spinner("Consultando seus dados..."):
                _send(session, thread_id, prompt)
        st.rerun()


def _send(session: AgentSession, thread_id: str, prompt: str) -> None:
    try:
        result = session.send(thread_id, prompt)
    except PendingApprovalError as exc:
        st.session_state["chat_error"] = str(exc)
        return
    if result.error:
        st.session_state["chat_error"] = result.error


def _show_chat_error() -> None:
    message = st.session_state.pop("chat_error", None)
    if message:
        st.error(message)


def _render_entry(entry: ChatEntry) -> None:
    with st.chat_message(entry.role):
        if entry.has_simulation:
            st.warning(SIMULATION_DISCLAIMER, icon="🧪")
        if entry.text:
            st.markdown(escape_markdown(entry.text))
        if entry.tool_events:
            _render_tool_events(entry.tool_events)


def _render_tool_events(events: list[ToolEvent]) -> None:
    """Mostra só o que é observável: tool, argumentos, resultado e erros."""
    with st.expander(f"🔧 Ferramentas usadas ({len(events)})"):
        for event in events:
            icon, label = _STATUS[event.status]
            st.markdown(f"{icon} `{event.name}` · {label}")
            if isinstance(event.result, dict) and event.result.get("replayed"):
                st.caption("Operação já aplicada anteriormente: nada foi gravado de novo.")
            st.caption("Argumentos")
            st.json(event.args, expanded=False)
            if event.result is not None:
                st.caption("Resultado")
                if isinstance(event.result, dict):
                    st.json(event.result, expanded=False)
                else:
                    st.code(str(event.result), language=None, wrap_lines=True)


def _render_approval(session: AgentSession, thread_id: str, pending: PendingApproval) -> None:
    with st.container(border=True):
        st.markdown("**Confirme a operação**")
        for action in pending.actions:
            st.markdown(escape_markdown(action.description))
        # A chave contém o interrupt_id: cada confirmação tem widgets próprios.
        key = pending.interrupt_id
        reason = st.text_input(
            "Motivo da rejeição (opcional)",
            placeholder="Ex.: o valor foi 80, não 85",
            key=f"reason_{key}",
        )
        approve_col, reject_col = st.columns(2)
        approved = approve_col.button("Aprovar", type="primary", width="stretch", key=f"approve_{key}")
        rejected = reject_col.button("Rejeitar", width="stretch", key=f"reject_{key}")

    if not (approved or rejected):
        return
    try:
        with st.spinner("Executando..." if approved else "Registrando sua decisão..."):
            result = session.decide(thread_id, key, approve=approved, reason=reason or None)
    except StaleApprovalError as exc:
        st.session_state["flash"] = str(exc)
    else:
        if result.error:
            st.session_state["chat_error"] = result.error
    st.rerun()
