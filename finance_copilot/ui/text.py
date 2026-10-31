"""Utilitários de texto para a interface."""


def escape_markdown(text: str) -> str:
    """Escapa cifrões: no markdown do Streamlit, "R$ 85 e R$ 90" vira uma fórmula LaTeX."""
    return text.replace("$", "\\$")
