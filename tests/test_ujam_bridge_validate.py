from __future__ import annotations

import textwrap

from pathlib import Path

import tests._stubs  # noqa: F401  # ensure pretty_midi stub
from tools.ujam_bridge import validate  # type: ignore


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_duplicate_and_range(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        product: demo
        octave_base: 0
        keyswitches:
          - name: a
            note: 10
            hold: true
          - name: b
            note: 10
            hold: false
        """,
    )
    issues = validate.validate(path)
    assert any("duplicate note" in m for m in issues)


def test_out_of_range(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        product: demo
        octave_base: 0
        keyswitches:
          - name: a
            note: 200
            hold: true
        """,
    )
    issues = validate.validate(path)
    assert any("out of range" in m for m in issues)


def test_schema_missing_hold(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        product: demo
        octave_base: 0
        keyswitches:
          - name: a
            note: 20
        """,
    )
    issues = validate.validate(path)
    assert any("missing hold" in m for m in issues)
