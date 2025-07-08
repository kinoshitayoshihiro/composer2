from transformer.sax_tokenizer import SaxTokenizer


def test_round_trip_with_artic() -> None:
    tok = SaxTokenizer()
    events = [
        {"bar": 0, "note": 60, "artic": "slide_up"},
        {"bar": 1, "note": 61, "artic": "slide_down"},
        {"bar": 2, "note": 62, "artic": "alt_hit"},
    ]
    ids = tok.encode(events)
    decoded = tok.decode(ids)
    assert decoded == events


def test_special_tokens_exist() -> None:
    tok = SaxTokenizer()
    assert "<SLIDE_UP>" in tok.vocab
    assert "<SLIDE_DOWN>" in tok.vocab
    assert "<ALT_HIT>" in tok.vocab
