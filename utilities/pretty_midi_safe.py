from __future__ import annotations

import os
import math
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
        midi = mido.MidiFile(tmp_path)
        try:
            from mido import MetaMessage, MidiTrack, bpm2tempo
        except Exception:  # pragma: no cover - optional dependency subset missing
            return midi

        initial_bpm: float | None = None
        try:
            _, tempi = pm.get_tempo_changes()
            if len(tempi) > 0:
                initial_bpm = float(tempi[0])
        except Exception:
            initial_bpm = None
        if not initial_bpm or not math.isfinite(initial_bpm) or initial_bpm <= 0:
            initial_bpm = float(getattr(pm, "initial_tempo", 120.0) or 120.0)

        tempo_value = bpm2tempo(initial_bpm)
        if midi.tracks:
            track0 = midi.tracks[0]
        else:  # pragma: no cover - ensure track exists
            track0 = MidiTrack()
            midi.tracks.append(track0)

        # Ensure track0 has a tempo marker
        if not any(getattr(msg, "type", "") == "set_tempo" for msg in track0):
            track0.insert(0, MetaMessage("set_tempo", tempo=tempo_value, time=0))

        def _meta_exists(track, kind: str) -> bool:
            return any(getattr(msg, "type", "") == kind for msg in track)

        def _first_meta(kind: str):
            for tr in midi.tracks:
                for msg in tr:
                    if getattr(msg, "type", "") == kind:
                        return msg
            return None

        need_meta_track = len(midi.tracks) < 2
        if not need_meta_track:
            track1 = midi.tracks[1]
            has_name = _meta_exists(track1, "track_name")
            has_time_sig = _meta_exists(track1, "time_signature")
            has_tempo = _meta_exists(track1, "set_tempo")
            need_meta_track = not (has_name and has_time_sig)
            if not need_meta_track and not has_tempo:
                track1.insert(0, MetaMessage("set_tempo", tempo=tempo_value, time=0))

        if need_meta_track:
            meta_track = MidiTrack()
            meta_track.append(MetaMessage("set_tempo", tempo=tempo_value, time=0))
            name_msg = _first_meta("track_name")
            if name_msg is not None:
                meta_track.append(name_msg.copy(time=0))  # type: ignore[attr-defined]
            else:
                meta_track.append(MetaMessage("track_name", name="metadata", time=0))

            time_sig_msg = _first_meta("time_signature")
            if time_sig_msg is not None:
                meta_track.append(time_sig_msg.copy(time=0))  # type: ignore[attr-defined]
            else:
                meta_track.append(
                    MetaMessage(
                        "time_signature",
                        numerator=4,
                        denominator=4,
                        clocks_per_click=24,
                        notated_32nd_notes_per_beat=8,
                        time=0,
                    )
                )
            meta_track.append(MetaMessage("end_of_track", time=0))
            insert_idx = 1 if midi.tracks else 0
            midi.tracks.insert(insert_idx, meta_track)
        return midi
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
