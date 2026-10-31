"""Acesso ao SQLite. Todas as consultas são parametrizadas; não há SQL vindo de fora."""

import sqlite3
from datetime import date

from finance_copilot.database import utc_now_iso
from finance_copilot.schemas import TransactionFilter

_TRANSACTION_COLUMNS = """
    t.id, t.kind, t.amount_cents, t.description, c.name AS category, t.occurred_on,
    t.source, t.created_at, t.updated_at
"""
_ACTIVE_JOIN = "FROM transactions t JOIN categories c ON c.id = t.category_id"


class Repository:
    """Recebe a conexão para participar da transação aberta pelo serviço."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- Categorias ---------------------------------------------------------------------

    def list_categories(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT id, name, kind FROM categories ORDER BY kind, id").fetchall()

    # --- Transações ---------------------------------------------------------------------

    def insert_transaction(
        self,
        kind: str,
        amount_cents: int,
        description: str,
        category_id: int,
        occurred_on: date,
        source: str,
    ) -> int:
        now = utc_now_iso()
        cursor = self.conn.execute(
            """INSERT INTO transactions
               (kind, amount_cents, description, category_id, occurred_on, source,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (kind, amount_cents, description, category_id, occurred_on.isoformat(), source, now, now),
        )
        return int(cursor.lastrowid)

    def get_transaction(self, transaction_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            f"""SELECT {_TRANSACTION_COLUMNS} {_ACTIVE_JOIN}
                WHERE t.id = ? AND t.deleted_at IS NULL""",
            (transaction_id,),
        ).fetchone()

    def update_transaction(
        self,
        transaction_id: int,
        kind: str,
        amount_cents: int,
        description: str,
        category_id: int,
        occurred_on: date,
    ) -> None:
        self.conn.execute(
            """UPDATE transactions
               SET kind = ?, amount_cents = ?, description = ?, category_id = ?,
                   occurred_on = ?, updated_at = ?
               WHERE id = ? AND deleted_at IS NULL""",
            (
                kind,
                amount_cents,
                description,
                category_id,
                occurred_on.isoformat(),
                utc_now_iso(),
                transaction_id,
            ),
        )

    def soft_delete_transaction(self, transaction_id: int) -> None:
        now = utc_now_iso()
        self.conn.execute(
            "UPDATE transactions SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (now, now, transaction_id),
        )

    @staticmethod
    def _filter_clause(filters: TransactionFilter, category_id: int | None) -> tuple[str, list]:
        clauses = ["t.deleted_at IS NULL"]
        params: list = []
        if filters.start_date:
            clauses.append("t.occurred_on >= ?")
            params.append(filters.start_date.isoformat())
        if filters.end_date:
            clauses.append("t.occurred_on <= ?")
            params.append(filters.end_date.isoformat())
        if category_id is not None:
            clauses.append("t.category_id = ?")
            params.append(category_id)
        if filters.kind:
            clauses.append("t.kind = ?")
            params.append(filters.kind)
        if filters.search:
            clauses.append("t.description LIKE ? ESCAPE '\\'")
            escaped = filters.search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        return " AND ".join(clauses), params

    def list_transactions(
        self, filters: TransactionFilter, category_id: int | None
    ) -> list[sqlite3.Row]:
        where, params = self._filter_clause(filters, category_id)
        return self.conn.execute(
            f"""SELECT {_TRANSACTION_COLUMNS} {_ACTIVE_JOIN} WHERE {where}
                ORDER BY t.occurred_on DESC, t.id DESC LIMIT ?""",
            (*params, filters.limit),
        ).fetchall()

    def transaction_totals(
        self, filters: TransactionFilter, category_id: int | None
    ) -> list[sqlite3.Row]:
        where, params = self._filter_clause(filters, category_id)
        return self.conn.execute(
            f"""SELECT t.kind, COUNT(*) AS count, SUM(t.amount_cents) AS total
                {_ACTIVE_JOIN} WHERE {where} GROUP BY t.kind""",
            params,
        ).fetchall()

    def find_identical(
        self, kind: str, amount_cents: int, occurred_on: date, description: str
    ) -> list[sqlite3.Row]:
        """Lançamentos iguais, usado apenas para alertar sobre duplicidade na confirmação."""
        return self.conn.execute(
            f"""SELECT {_TRANSACTION_COLUMNS} {_ACTIVE_JOIN}
                WHERE t.deleted_at IS NULL AND t.kind = ? AND t.amount_cents = ?
                  AND t.occurred_on = ? AND lower(t.description) = lower(?)""",
            (kind, amount_cents, occurred_on.isoformat(), description),
        ).fetchall()

    # --- Agregações ---------------------------------------------------------------------

    def totals_by_kind(self, start: date, end: date) -> dict[str, tuple[int, int]]:
        """Retorna {kind: (total_cents, count)} para o intervalo de datas."""
        rows = self.conn.execute(
            """SELECT kind, COALESCE(SUM(amount_cents), 0) AS total, COUNT(*) AS count
               FROM transactions
               WHERE deleted_at IS NULL AND occurred_on BETWEEN ? AND ?
               GROUP BY kind""",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return {row["kind"]: (row["total"], row["count"]) for row in rows}

    def totals_by_category(self, start: date, end: date, kind: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"""SELECT c.name AS category, SUM(t.amount_cents) AS total, COUNT(*) AS count
                {_ACTIVE_JOIN}
                WHERE t.deleted_at IS NULL AND t.kind = ? AND t.occurred_on BETWEEN ? AND ?
                GROUP BY c.name
                ORDER BY total DESC, c.name""",
            (kind, start.isoformat(), end.isoformat()),
        ).fetchall()

    # --- Salário ------------------------------------------------------------------------

    def salary_for_month(self, year_month: str) -> int | None:
        row = self.conn.execute(
            """SELECT amount_cents FROM monthly_salary
               WHERE effective_from <= ? ORDER BY effective_from DESC LIMIT 1""",
            (year_month,),
        ).fetchone()
        return row["amount_cents"] if row else None

    def upsert_salary(self, effective_from: str, amount_cents: int) -> None:
        self.conn.execute(
            """INSERT INTO monthly_salary (effective_from, amount_cents, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT (effective_from)
               DO UPDATE SET amount_cents = excluded.amount_cents, updated_at = excluded.updated_at""",
            (effective_from, amount_cents, utc_now_iso()),
        )

    # --- Idempotência -------------------------------------------------------------------

    def get_applied_operation(self, idempotency_key: str) -> str | None:
        row = self.conn.execute(
            "SELECT result_json FROM applied_operations WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return row["result_json"] if row else None

    def save_applied_operation(self, idempotency_key: str, operation: str, result_json: str) -> None:
        self.conn.execute(
            """INSERT INTO applied_operations (idempotency_key, operation, result_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (idempotency_key, operation, result_json, utc_now_iso()),
        )
