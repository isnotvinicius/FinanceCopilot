"""Política de aprovação humana das operações de escrita (HumanInTheLoopMiddleware).

O texto da confirmação é gerado aqui, a partir dos argumentos reais da tool e do banco. Assim, o
usuário aprova o que será executado, e não a forma como o modelo descreveu a ação.
"""

import logging

from langchain.agents.middleware import InterruptOnConfig
from pydantic import ValidationError

from finance_copilot.agent.tool_schemas import (
    CreateTransactionArgs,
    UpdateTransactionArgs,
)
from finance_copilot.agent.tools import WRITE_TOOLS
from finance_copilot.errors import FinanceError
from finance_copilot.money import format_brl, to_cents
from finance_copilot.schemas import TransactionInput, TransactionView
from finance_copilot.service import FinanceService

logger = logging.getLogger(__name__)

_KIND_LABELS = {"expense": "despesa", "income": "receita adicional"}
_WRITE_SCHEMAS = {tool.name: tool.args_schema for tool in WRITE_TOOLS}


def _format_transaction(item: TransactionView) -> str:
    return (
        f"#{item.id} · {_KIND_LABELS[item.kind]} · {item.amount.brl} · {item.description} · "
        f"{item.category} · {item.occurred_on:%d/%m/%Y}"
    )


def _describe_create(args: CreateTransactionArgs, service: FinanceService) -> str:
    data = TransactionInput(
        kind=args.kind,
        amount_cents=to_cents(args.amount),
        description=args.description,
        category=args.category,
        occurred_on=args.occurred_on,
    )
    lines = [
        f"**Registrar {_KIND_LABELS[args.kind]}**",
        f"- Valor: **{format_brl(data.amount_cents)}**",
        f"- Descrição: {args.description}",
        f"- Categoria: {args.category}",
        f"- Data: {args.occurred_on:%d/%m/%Y}",
    ]
    try:
        service.validate_transaction(data)
    except FinanceError as exc:
        lines.append(f"\n⚠️ Esta operação será recusada: {exc}")
    identical = service.find_identical_transactions(data)
    if identical:
        lines.append(f"\n⚠️ Já existe um lançamento igual: {_format_transaction(identical[0])}")
    return "\n".join(lines)


def _describe_update(args: UpdateTransactionArgs, service: FinanceService) -> str:
    try:
        current = service.get_transaction(args.transaction_id)
    except FinanceError as exc:
        return f"**Alterar transação**\n\n⚠️ Esta operação será recusada: {exc}"
    changes = [("Tipo", _KIND_LABELS[current.kind], _KIND_LABELS.get(args.kind))]
    changes.append(
        ("Valor", current.amount.brl, format_brl(to_cents(args.amount)) if args.amount else None)
    )
    changes.append(("Descrição", current.description, args.description))
    changes.append(("Categoria", current.category, args.category))
    changes.append(
        (
            "Data",
            f"{current.occurred_on:%d/%m/%Y}",
            f"{args.occurred_on:%d/%m/%Y}" if args.occurred_on else None,
        )
    )
    lines = [f"**Alterar transação #{current.id}** ({current.description})"]
    lines += [f"- {label}: {old} → **{new}**" for label, old, new in changes if new]
    return "\n".join(lines)


def _describe_delete(transaction_id: int, service: FinanceService) -> str:
    try:
        current = service.get_transaction(transaction_id)
    except FinanceError as exc:
        return f"**Excluir transação**\n\n⚠️ Esta operação será recusada: {exc}"
    return f"**Excluir transação**\n- {_format_transaction(current)}"


def _describe_salary(cents: int, month: str, service: FinanceService) -> str:
    current = service.get_salary(month)
    return (
        "**Definir salário mensal**\n"
        f"- Atual: {current.brl if current else 'não configurado'}\n"
        f"- Novo: **{format_brl(cents)}**, a partir de {month}"
    )


def describe_tool_call(tool_call, state, runtime) -> str:
    """Gera o texto do cartão de confirmação. Apenas lê dados e nunca interrompe o fluxo."""
    name, args = tool_call["name"], tool_call["args"]
    service = runtime.context.service
    try:
        parsed = _WRITE_SCHEMAS[name].model_validate(args)
        if name == "create_transaction":
            return _describe_create(parsed, service)
        if name == "update_transaction":
            return _describe_update(parsed, service)
        if name == "delete_transaction":
            return _describe_delete(parsed.transaction_id, service)
        month = parsed.effective_from or service.current_month()
        return _describe_salary(to_cents(parsed.amount), month, service)
    except Exception:
        logger.exception("Falha ao descrever a operação %s", name)
        return f"**{name}**\n\nArgumentos: `{args}`"


def _has_valid_args(request) -> bool:
    """Não pede aprovação para o que não pode executar.

    Argumentos inválidos seguem direto para a tool, que aplica este mesmo schema e devolve o erro
    ao modelo sem gravar nada. É uma função pura dos argumentos: não há como um estado alterado
    fazer uma operação passar sem aprovação.
    """
    call = request.tool_call
    try:
        _WRITE_SCHEMAS[call["name"]].model_validate(call["args"])
    except ValidationError:
        return False
    return True


def build_interrupt_on() -> dict[str, InterruptOnConfig]:
    """Toda tool de WRITE_TOOLS é coberta; um teste garante que nenhuma fique de fora."""
    return {
        tool.name: InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=describe_tool_call,
            when=_has_valid_args,
        )
        for tool in WRITE_TOOLS
    }
