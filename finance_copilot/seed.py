"""Popula o banco com dados FICTÍCIOS do mês atual e do mês anterior.

Uso: python -m finance_copilot.seed

É reproduzível: cada linha tem uma chave de idempotência fixa (mês + posição), portanto rodar
de novo não duplica nada. No mês atual, datas que ainda não chegaram são trazidas para hoje.
"""

from datetime import date

from finance_copilot.config import get_settings
from finance_copilot.database import Database
from finance_copilot.schemas import TransactionInput
from finance_copilot.service import FinanceService, month_bounds, shift_month

SALARY_CENTS = 6_500_00

# (dia, tipo, centavos, descrição, categoria)
PREVIOUS_MONTH = [
    (2, "expense", 1_800_00, "Aluguel", "Moradia"),
    (3, "expense", 412_30, "Supermercado", "Alimentação"),
    (5, "expense", 89_90, "Plano de internet", "Assinaturas"),
    (8, "expense", 152_00, "Combustível", "Transporte"),
    (11, "expense", 96_50, "Restaurante japonês", "Alimentação"),
    (14, "expense", 180_00, "Show de música", "Lazer"),
    (18, "expense", 240_00, "Consulta médica", "Saúde"),
    (20, "income", 800_00, "Projeto freelance de site", "Freelance"),
    (22, "expense", 318_70, "Supermercado", "Alimentação"),
    (26, "expense", 120_00, "Cinema e jantar", "Lazer"),
]
CURRENT_MONTH = [
    (1, "expense", 1_800_00, "Aluguel", "Moradia"),
    (2, "expense", 465_80, "Supermercado", "Alimentação"),
    (4, "expense", 89_90, "Plano de internet", "Assinaturas"),
    (6, "expense", 210_00, "Combustível", "Transporte"),
    (9, "expense", 143_00, "Restaurante italiano", "Alimentação"),
    (12, "expense", 320_00, "Viagem de fim de semana", "Lazer"),
    (15, "expense", 259_90, "Tênis de corrida", "Compras"),
    (17, "income", 450_00, "Dividendos", "Investimentos"),
    (19, "expense", 377_40, "Supermercado", "Alimentação"),
    (23, "expense", 199_00, "Curso online", "Educação"),
    (25, "expense", 150_00, "Bar com amigos", "Lazer"),
]


def seed(service: FinanceService) -> int:
    """Insere o que faltar e retorna quantas transações novas foram criadas."""
    today = service.today()
    current = service.current_month()
    previous = shift_month(current, -1)

    service.set_monthly_salary(
        SALARY_CENTS, effective_from=previous, idempotency_key="seed:salary:v1"
    )
    created = 0
    for month, rows in ((previous, PREVIOUS_MONTH), (current, CURRENT_MONTH)):
        last_day = month_bounds(month)[1]
        limit = today if month == current else last_day
        for position, (day, kind, cents, description, category) in enumerate(rows):
            occurred_on = min(date(last_day.year, last_day.month, day), limit)
            result = service.create_transaction(
                TransactionInput(
                    kind=kind,
                    amount_cents=cents,
                    description=description,
                    category=category,
                    occurred_on=occurred_on,
                ),
                idempotency_key=f"seed:{month}:{position}",
                source="seed",
            )
            created += not result.replayed
    return created


def main() -> None:
    settings = get_settings()
    created = seed(FinanceService(Database(settings.finance_db_path)))
    print(f"Seed concluído em {settings.finance_db_path}: {created} transação(ões) nova(s).")


if __name__ == "__main__":
    main()
