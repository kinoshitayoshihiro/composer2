import pytest
from generator.chord_voicer import parse_chord_symbol

@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("Cadd9", ["C", "E", "G", "D"]),
        ("Dsus2", ["D", "E", "A"]),
        ("Am7", ["A", "C", "E", "G"]),
        ("F6", ["F", "A", "C", "D"]),
    ],
)
def test_parse_chord_symbol(symbol, expected):
    notes = parse_chord_symbol(symbol)
    names = [p.name for p in notes]
    assert names == expected
