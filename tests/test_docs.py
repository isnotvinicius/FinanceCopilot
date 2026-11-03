"""A documentação cita arquivos e símbolos reais: este teste falha se uma referência envelhecer."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PATH_PATTERN = re.compile(r"(?:finance_copilot|tests)/[\w/.]+\.(?:py|toml)|\.env\.example")
# `arquivo.py::simbolo` nos documentos
SYMBOL_PATTERN = re.compile(r"((?:finance_copilot|tests)/[\w/.]+\.py)::(\w+)")


def test_expected_documents_exist():
    assert {doc.name for doc in DOCS} == {
        "README.md",
        "ARCHITECTURE.md",
        "LANGCHAIN_GUIDE.md",
        "INTERVIEW.md",
        "LLM_PROVIDERS.md",
    }


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_referenced_paths_exist(doc):
    missing = [p for p in set(PATH_PATTERN.findall(doc.read_text())) if not (ROOT / p).exists()]
    assert not missing


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_referenced_symbols_exist(doc):
    missing = [
        f"{path}::{symbol}"
        for path, symbol in SYMBOL_PATTERN.findall(doc.read_text())
        if not re.search(rf"\b{symbol}\b", (ROOT / path).read_text())
    ]
    assert not missing
