"""Contexto de execução do agente (`context_schema` do create_agent)."""

from dataclasses import dataclass

from finance_copilot.service import FinanceService


@dataclass(frozen=True)
class AgentContext:
    """Dependências entregues a tools e middlewares a cada execução.

    Diferente do state, o contexto não entra no checkpoint nem é visto pelo modelo. É a forma
    de injetar o serviço sem variáveis globais, o que permite usar um banco temporário nos testes.
    """

    service: FinanceService
