import warnings
from transformer.tokenizer_piano import PianoTokenizer


def test_warn_on_unknown():
    tok = PianoTokenizer()
    events = [{"bar": 0, "note": 999, "hand": "lh"}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok.encode(events)
        assert any("unk token rate" in str(wi.message) for wi in w)


def test_warn_on_unknown_sax_artic() -> None:
    from transformer.sax_tokenizer import SaxTokenizer

    tok = SaxTokenizer()
    events = [{"bar": 0, "note": 60, "artic": "growl"}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok.encode(events)
        assert any("unk token rate" in str(wi.message) for wi in w)
