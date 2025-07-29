import importlib.util
import pytest

transformers_available = importlib.util.find_spec("transformers") is not None
peft_available = importlib.util.find_spec("peft") is not None


@pytest.mark.skipif(not (transformers_available and peft_available), reason="missing transformers or peft")
def test_sax_transformer_params() -> None:
    from transformer.sax_transformer import SaxTransformer

    model = SaxTransformer(vocab_size=100)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    lora_params = [n for n, _ in model.named_parameters() if "lora" in n]
    assert lora_params


@pytest.mark.skipif(not (transformers_available and peft_available), reason="missing transformers or peft")
def test_sax_transformer_invalid_vocab() -> None:
    from transformer.sax_transformer import SaxTransformer

    with pytest.raises(ValueError):
        SaxTransformer(vocab_size=0)
