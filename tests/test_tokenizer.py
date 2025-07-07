from transformer.tokenizer_piano import PianoTokenizer


def test_round_trip_simple() -> None:
    tokenizer = PianoTokenizer()
    events = [
        {"bar": 0, "note": 60, "hand": "lh"},
        {"bar": 1, "note": 64, "hand": "rh"},
    ]
    tokens = tokenizer.encode(events)
    decoded = tokenizer.decode(tokens)
    assert decoded == events
