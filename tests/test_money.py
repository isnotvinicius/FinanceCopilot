from decimal import Decimal

import pytest

from finance_copilot.money import apply_percent, format_brl, parse_money, percent_of, to_cents


@pytest.mark.parametrize(
    ("raw", "expected_cents"),
    [
        (85, 8500),
        (85.5, 8550),
        (19.99, 1999),  # em float puro, 19.99 * 100 == 1998.9999999999998
        (0.1, 10),
        ("85,50", 8550),
        ("R$ 1.234,56", 123456),
        ("1.234", 123400),
        ("1234.56", 123456),
    ],
)
def test_parse_money_accepts_common_formats(raw, expected_cents):
    assert to_cents(parse_money(raw)) == expected_cents


@pytest.mark.parametrize("raw", ["abc", "", True, None, float("nan"), float("inf"), 10.123, "1,234"])
def test_parse_money_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_money(raw)


def test_format_brl():
    assert format_brl(123456) == "R$ 1.234,56"
    assert format_brl(5) == "R$ 0,05"
    assert format_brl(-8500) == "-R$ 85,00"
    assert format_brl(100000000) == "R$ 1.000.000,00"


def test_percent_helpers():
    assert apply_percent(30000, Decimal(-20)) == -6000
    assert apply_percent(1001, Decimal(50)) == 501  # arredonda meio centavo para cima
    assert percent_of(1, 3) == Decimal("33.3")
    assert percent_of(1, 0) is None
