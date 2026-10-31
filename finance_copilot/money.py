"""Conversão e formatação de dinheiro.

Toda a aplicação calcula em centavos inteiros. `Decimal` só aparece na fronteira de entrada
(texto do usuário ou JSON da LLM) e em porcentagens. `float` nunca é usado em cálculo: quando
um JSON traz um número, ele é convertido via `str()`, preservando o valor escrito.
"""

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

MAX_AMOUNT_CENTS = 100_000_000_00  # R$ 100 milhões: limite de sanidade para o MVP

_THOUSANDS_ONLY = re.compile(r"^\d{1,3}(\.\d{3})+$")


def parse_money(value: object) -> Decimal:
    """Converte a entrada em Decimal de reais; aceita 85, 85.5, "85,50" e "R$ 1.234,56"."""
    if isinstance(value, bool):
        raise ValueError("valor monetário inválido")
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, (int, float)):
        amount = _to_decimal(str(value))
    elif isinstance(value, str):
        amount = _to_decimal(_normalize_brl_text(value))
    else:
        raise ValueError("valor monetário inválido")
    if not amount.is_finite():
        raise ValueError("valor monetário inválido")
    if amount != amount.quantize(Decimal("0.01")):
        raise ValueError("valor monetário com mais de duas casas decimais")
    return amount


def _normalize_brl_text(text: str) -> str:
    cleaned = text.replace("R$", "").replace(" ", "").strip()
    if "," in cleaned:
        return cleaned.replace(".", "").replace(",", ".")
    if _THOUSANDS_ONLY.match(cleaned):
        return cleaned.replace(".", "")
    return cleaned


def _to_decimal(text: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("valor monetário inválido") from exc


def to_cents(amount: Decimal) -> int:
    return int((amount * 100).to_integral_value(rounding=ROUND_HALF_UP))


def format_brl(cents: int) -> str:
    """Formata centavos no padrão brasileiro: 123456 -> 'R$ 1.234,56'."""
    sign = "-" if cents < 0 else ""
    reais, centavos = divmod(abs(cents), 100)
    return f"{sign}R$ {reais:,}".replace(",", ".") + f",{centavos:02d}"


def apply_percent(cents: int, percent: Decimal) -> int:
    """Retorna `percent`% de `cents`, arredondado para o centavo."""
    result = Decimal(cents) * percent / 100
    return int(result.to_integral_value(rounding=ROUND_HALF_UP))


def percent_of(part: int, whole: int) -> Decimal | None:
    """Razão em porcentagem com 1 casa; None quando a base é zero (não há razão definida)."""
    if whole == 0:
        return None
    return (Decimal(part) * 100 / Decimal(whole)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
