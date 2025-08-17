from __future__ import annotations

import os
import tempfile
from typing import Any

import pretty_midi


def pm_to_mido(pm: pretty_midi.PrettyMIDI):
    """Return a :class:`mido.MidiFile` for *pm* using a temporary file.

    The conversion writes ``pm`` to a real temporary ``.mid`` file and loads it
    via :mod:`mido`. ``ImportError`` is raised when :mod:`mido` is missing. The
    temporary file is removed on success and best effort on failure.
    """
    try:
        import mido  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pm_to_mido requires 'mido' (pip install mido>=1.3)") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        if hasattr(pm, "write"):
            pm.write(tmp_path)
        else:  # pragma: no cover - extremely old pretty_midi
            raise RuntimeError("PrettyMIDI.write unavailable in this environment")
        return mido.MidiFile(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:  # pragma: no cover - best effort cleanup
            pass


def new_pm(*args: Any, **kwargs: Any) -> pretty_midi.PrettyMIDI:
    """Instantiate :class:`pretty_midi.PrettyMIDI` with a dummy ``midi_data``.

    The returned object mirrors :class:`pretty_midi.PrettyMIDI` but always
    provides a ``midi_data`` attribute set to ``None`` to avoid ``AttributeError``
    in older ``pretty_midi`` versions.
    """
    pm = pretty_midi.PrettyMIDI(*args, **kwargs)
    setattr(pm, "midi_data", None)
    return pm
