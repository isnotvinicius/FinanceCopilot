"""Preenchimento rápido do formulário manual com structured output.

Este fluxo é determinístico (texto -> rascunho -> formulário), portanto não usa o agente: não há
ferramentas a escolher nem ciclo de decisões. A extração nunca grava: o usuário revisa e salva.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finance_copilot.agent.prompts import render_extraction_prompt
from finance_copilot.agent.tool_schemas import TransactionDraft
from finance_copilot.service import FinanceService


def extract_transaction_draft(
    model: BaseChatModel, service: FinanceService, text: str
) -> TransactionDraft:
    extractor = model.with_structured_output(TransactionDraft)
    system = render_extraction_prompt(service.today(), service.list_categories())
    return extractor.invoke([SystemMessage(system), HumanMessage(text)])
