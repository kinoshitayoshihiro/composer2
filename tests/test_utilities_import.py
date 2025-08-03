import importlib


def test_utilities_import_smoke():
    assert importlib.import_module("utilities")
