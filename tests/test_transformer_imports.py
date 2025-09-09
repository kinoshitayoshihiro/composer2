import importlib

MODULES = [
    'transformer.piano_transformer',
    'transformer.tokenizer_piano',
    'transformer.sax_transformer',
    'transformer.sax_tokenizer',
]

def test_imports():
    for mod in MODULES:
        assert importlib.import_module(mod)
