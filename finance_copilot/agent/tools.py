"""Tools do agente: adaptadores finos entre o modelo e o FinanceService.

Cada tool recebe argumentos validados pelo schema, delega ao serviço e devolve um dict
estruturado. Nenhuma contém SQL ou regra de negócio. O docstring é o que a LLM lê para decidir
quando chamar a tool.

Exceções de domínio são convertidas em mensagens de erro para o modelo pelo ToolErrorMiddleware.
O parâmetro `runtime` é injetado pelo LangGraph e não aparece no schema enviado ao modelo.
"""

from langchain.tools import ToolRuntime, tool

from finance_copilot.agent.context import AgentContext
from finance_copilot.agent.tool_schemas import (
    CategorySummaryArgs,
    ComparePeriodsArgs,
    CreateTransactionArgs,
    DeleteTransactionArgs,
    GetTransactionsArgs,
    PeriodArgs,
    SetMonthlySalaryArgs,
    SimulateBudgetArgs,
    UpdateTransactionArgs,
)
from finance_copilot.money import to_cents
from finance_copilot.schemas import TransactionChanges, TransactionFilter, TransactionInput


def _idempotency_key(runtime: ToolRuntime[AgentContext]) -> str:
    """Identifica a chamada da tool feita pelo modelo.

    O `tool_call_id` fica salvo no checkpoint, dentro da AIMessage, então não muda quando a
    execução é retomada ou repetida. É isso que torna a gravação segura contra replays.
    """
    thread_id = runtime.config["configurable"]["thread_id"]
    return f"agent:{thread_id}:{runtime.tool_call_id}"


# --- Escrita: exigem aprovação humana (ver approvals.py) ------------------------------------


@tool(args_schema=CreateTransactionArgs)
def create_transaction(
    kind, amount, description, category, occurred_on, runtime: ToolRuntime[AgentContext]
) -> dict:
    """Registra uma despesa ou uma receita adicional. Não use para salário."""
    result = runtime.context.service.create_transaction(
        TransactionInput(
            kind=kind,
            amount_cents=to_cents(amount),
            description=description,
            category=category,
            occurred_on=occurred_on,
        ),
        idempotency_key=_idempotency_key(runtime),
        source="agent",
    )
    return result.model_dump(mode="json")


@tool(args_schema=UpdateTransactionArgs)
def update_transaction(
    transaction_id,
    runtime: ToolRuntime[AgentContext],
    kind=None,
    amount=None,
    description=None,
    category=None,
    occurred_on=None,
) -> dict:
    """Altera campos de uma transação existente. Envie apenas os campos que mudam."""
    result = runtime.context.service.update_transaction(
        transaction_id,
        TransactionChanges(
            kind=kind,
            amount_cents=to_cents(amount) if amount is not None else None,
            description=description,
            category=category,
            occurred_on=occurred_on,
        ),
        idempotency_key=_idempotency_key(runtime),
    )
    return result.model_dump(mode="json")


@tool(args_schema=DeleteTransactionArgs)
def delete_transaction(transaction_id, runtime: ToolRuntime[AgentContext]) -> dict:
    """Exclui uma transação existente pelo id."""
    result = runtime.context.service.delete_transaction(
        transaction_id, idempotency_key=_idempotency_key(runtime)
    )
    return result.model_dump(mode="json")


@tool(args_schema=SetMonthlySalaryArgs)
def set_monthly_salary(amount, runtime: ToolRuntime[AgentContext], effective_from=None) -> dict:
    """Define o salário mensal recorrente, válido a partir de um mês. Meses anteriores não mudam."""
    result = runtime.context.service.set_monthly_salary(
        to_cents(amount),
        effective_from=effective_from,
        idempotency_key=_idempotency_key(runtime),
    )
    return result.model_dump(mode="json")


# --- Leitura: executam sem aprovação --------------------------------------------------------


@tool(args_schema=GetTransactionsArgs)
def get_transactions(
    runtime: ToolRuntime[AgentContext],
    start_date=None,
    end_date=None,
    category=None,
    kind=None,
    search=None,
    limit=20,
) -> dict:
    """Lista movimentações com filtros. Use para ver lançamentos e descobrir ids. O salário não
    aparece aqui. Inclui totais de tudo que atende ao filtro, mesmo além do limite."""
    page = runtime.context.service.list_transactions(
        TransactionFilter(
            start_date=start_date,
            end_date=end_date,
            category=category,
            kind=kind,
            search=search,
            limit=limit,
        )
    )
    return page.model_dump(mode="json")


@tool(args_schema=PeriodArgs)
def get_financial_summary(start_month, runtime: ToolRuntime[AgentContext], end_month=None) -> dict:
    """Retorna salário, receitas adicionais, receita total, despesas e saldo de meses inteiros.
    Use para perguntas sobre saldo ou dinheiro disponível."""
    summary = runtime.context.service.financial_summary(start_month, end_month)
    return summary.model_dump(mode="json")


@tool(args_schema=CategorySummaryArgs)
def get_category_summary(
    start_month, runtime: ToolRuntime[AgentContext], end_month=None, kind="expense"
) -> dict:
    """Retorna totais por categoria, do maior para o menor, com participação percentual. Use para
    saber quanto foi gasto em uma categoria ou qual é a maior despesa."""
    summary = runtime.context.service.category_summary(start_month, end_month, kind)
    return summary.model_dump(mode="json")


@tool(args_schema=ComparePeriodsArgs)
def compare_periods(
    baseline_start,
    target_start,
    runtime: ToolRuntime[AgentContext],
    baseline_end=None,
    target_end=None,
    category=None,
) -> dict:
    """Compara dois períodos: receitas, despesas, saldo e diferença por categoria, com a
    contribuição de cada uma para o aumento. Use para 'mais que', 'aumento', 'mês passado vs atual'."""
    comparison = runtime.context.service.compare_periods(
        baseline_start,
        target_start,
        baseline_end=baseline_end,
        target_end=target_end,
        category=category,
    )
    return comparison.model_dump(mode="json")


@tool(args_schema=SimulateBudgetArgs)
def simulate_budget(
    runtime: ToolRuntime[AgentContext],
    base_month=None,
    adjustments=None,
    extra_monthly_saving=None,
    months=1,
) -> dict:
    """Simula cenários hipotéticos sem gravar nada: variação percentual de despesas e/ou economia
    mensal acumulada ao longo de vários meses. O resultado NÃO é um valor real."""
    simulation = runtime.context.service.simulate_budget(
        base_month=base_month,
        adjustments=adjustments,
        extra_monthly_saving_cents=(
            to_cents(extra_monthly_saving) if extra_monthly_saving is not None else None
        ),
        months=months,
    )
    return simulation.model_dump(mode="json")


WRITE_TOOLS = [create_transaction, update_transaction, delete_transaction, set_monthly_salary]
READ_TOOLS = [
    get_transactions,
    get_financial_summary,
    get_category_summary,
    compare_periods,
    simulate_budget,
]
ALL_TOOLS = [*READ_TOOLS, *WRITE_TOOLS]
