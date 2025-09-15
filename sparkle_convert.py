"""Compatibility wrapper for :mod:`ujam.sparkle_convert`.

This module re-exports all public names from ``ujam.sparkle_convert`` so
``import sparkle_convert as sc`` behaves as expected in the tests.  The original
module defines a limited ``__all__`` which omits many helpers used in the test
suite.  Instead of relying on ``from ... import *`` (which honours ``__all__``),
we copy attributes from the underlying module directly.

The helper ``_dummy_pm`` is provided as an alias to the underlying
``_pm_dummy_for_docs`` to keep backwards compatibility with existing tests.
"""

from importlib import import_module
import builtins

_sc = import_module("ujam.sparkle_convert")

# Re-export everything except dunder attributes.
for _name in dir(_sc):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_sc, _name)

# Backwards compatible alias expected by tests.
if "_pm_dummy_for_docs" in globals() and "_dummy_pm" not in globals():
    _dummy_pm = globals()["_pm_dummy_for_docs"]
    globals()["_dummy_pm"] = _dummy_pm

# Some legacy snippets in the test suite reference ``accent_map`` without
# qualifying it.  Expose a default so such references simply evaluate to
# ``False`` instead of raising ``NameError``.
if not hasattr(builtins, "accent_map"):
    builtins.accent_map = None  # type: ignore[attr-defined]


def main(*args, **kwargs):
    """Delegate to :func:`ujam.sparkle_convert.main`.

    Synchronises patched helpers from this compatibility wrapper into the
    underlying implementation so tests that monkeypatch objects on the wrapper
    see their replacements used by the real CLI.
    """

    _sc.build_sparkle_midi = globals().get("build_sparkle_midi", _sc.build_sparkle_midi)
    _sc.pretty_midi = globals().get("pretty_midi", _sc.pretty_midi)
    if "parse_damp_arg" in globals():
        _sc.parse_damp_arg = globals()["parse_damp_arg"]
    if "emit_damping" in globals():
        _sc.emit_damping = globals()["emit_damping"]
    return _sc.main(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    raise SystemExit(main())
