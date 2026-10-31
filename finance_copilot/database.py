"""Conexão e esquema do banco financeiro.

O histórico do agente fica em outro arquivo (`agent_state.db`), gerenciado pelo checkpointer
do LangGraph. Apagar a conversa nunca afeta o dinheiro, e vice-versa.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('expense', 'income'))
);

CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL CHECK (kind IN ('expense', 'income')),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    description  TEXT    NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES categories (id),
    occurred_on  TEXT    NOT NULL,
    source       TEXT    NOT NULL CHECK (source IN ('agent', 'manual', 'seed')),
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    deleted_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_transactions_occurred_on ON transactions (occurred_on);

-- O salário é configuração, não transação. Cada linha vale do mês indicado em diante, então
-- mudar o salário hoje não altera a receita dos meses anteriores.
CREATE TABLE IF NOT EXISTS monthly_salary (
    effective_from TEXT    PRIMARY KEY,
    amount_cents   INTEGER NOT NULL CHECK (amount_cents > 0),
    updated_at     TEXT    NOT NULL
);

-- Registro de toda escrita já aplicada. A chave primária é a última barreira contra duplicação.
CREATE TABLE IF NOT EXISTS applied_operations (
    idempotency_key TEXT PRIMARY KEY,
    operation       TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""

DEFAULT_CATEGORIES = [
    ("Alimentação", "expense"),
    ("Moradia", "expense"),
    ("Transporte", "expense"),
    ("Saúde", "expense"),
    ("Lazer", "expense"),
    ("Educação", "expense"),
    ("Assinaturas", "expense"),
    ("Compras", "expense"),
    ("Outras despesas", "expense"),
    ("Freelance", "income"),
    ("Investimentos", "income"),
    ("Outras receitas", "income"),
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transações são explícitas, em transaction().
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            conn.executemany(
                "INSERT OR IGNORE INTO categories (name, kind) VALUES (?, ?)", DEFAULT_CATEGORIES
            )
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transação de escrita. BEGIN IMMEDIATE serializa escritores concorrentes."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
