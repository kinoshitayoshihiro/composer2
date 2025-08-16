# Ensure sitecustomize is imported during tests even if sys.path/working dir differs.
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

pytest_plugins = ("tests._asyncio_plugin",)

import pytest
try:
    from music21 import instrument
except Exception:  # pragma: no cover - optional
    instrument = None


@pytest.fixture
def _strings_gen():
    """Fixture returning a basic StringsGenerator."""
    try:
        from generator.strings_generator import StringsGenerator
    except Exception:
        pytest.skip("strings generator not available")

    def _create_generator(**kwargs):
        defaults = {
            "global_settings": {},
            "default_instrument": instrument.Violin() if instrument else None,
            "part_name": "strings",
            "global_tempo": 120,
            "global_time_signature": "4/4",
            "global_key_signature_tonic": "C",
            "global_key_signature_mode": "major",
        }
        defaults.update(kwargs)
        return StringsGenerator(**defaults)

    return _create_generator
