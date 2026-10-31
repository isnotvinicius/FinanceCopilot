"""Fixtures comuns. Nenhum teste aqui acessa a rede nem exige credenciais."""

import os
from datetime import date

import pytest

from finance_copilot.database import Database
from finance_copilot.schemas import TransactionInput
from finance_copilot.service import FinanceService

os.environ["LANGSMITH_TRACING"] = "false"

TODAY = date(2026, 10, 15)


@pytest.fixture
def service(tmp_path) -> FinanceService:
    return FinanceService(Database(tmp_path / "finance.db"), clock=lambda: TODAY)


class Builder:
    """Cria transações de teste com chaves únicas, sem repetir campos irrelevantes."""

    def __init__(self, service: FinanceService) -> None:
        self.service = service
        self.counter = 0

    def add(self, cents, category="Alimentação", on=TODAY, kind="expense", description="Teste"):
        self.counter += 1
        return self.service.create_transaction(
            TransactionInput(
                kind=kind,
                amount_cents=cents,
                description=description,
                category=category,
                occurred_on=on,
            ),
            idempotency_key=f"test:{self.counter}",
            source="manual",
        ).transaction


@pytest.fixture
def builder(service) -> Builder:
    return Builder(service)
