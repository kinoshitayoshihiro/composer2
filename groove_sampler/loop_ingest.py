"""Loop ingestion for the groove sampler."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import pretty_midi

from utilities.drum_map_registry import GM_DRUM_MAP

PPQ = 480
RESOLUTION = 16

# mapping from MIDI pitch to drum label
_PITCH_TO_LABEL: dict[int, str] = {val[1]: k for k, val in GM_DRUM_MAP.items()}

State = tuple[int, str]
LoadResult = (
    tuple[
        list[tuple[list[State], str]],
        dict[str, float],
        dict[str, Counter[int]],
        dict[str, Counter[int]],
    ]
)


def _iter_midi(path: Path) -> Iterator[tuple[int, str, int, int]]:
    """Yield quantised events from a MIDI file."""

    pm = pretty_midi.PrettyMIDI(str(path))
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if tempo.size else 120.0
    sec_per_beat = 60.0 / bpm

    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            beat = note.start / sec_per_beat
            step = int(round(beat * (RESOLUTION / 4)))
            q_pos = step / (RESOLUTION / 4)
            micro = int(round((beat - q_pos) * PPQ))
            label = _PITCH_TO_LABEL.get(note.pitch, str(note.pitch))
            yield step, label, note.velocity, micro


def _iter_wav(path: Path) -> Iterator[tuple[int, str, int, int]]:
    """Return onset positions from a WAV file using librosa."""

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("WAV support requires librosa") from exc

    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo else 120.0
    sec_per_beat = 60.0 / bpm
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    for t in onsets:
        beat = t / sec_per_beat
        step = int(round(beat * (RESOLUTION / 4)))
        micro = 0
        yield step, "perc", 100, micro


def scan_loops(loop_dir: Path, exts: Sequence[str]) -> LoadResult:
    """Parse loops from ``loop_dir`` with given extensions."""

    events: list[tuple[list[State], str]] = []
    vel_map: dict[str, list[int]] = defaultdict(list)
    micro_map: dict[str, list[int]] = defaultdict(list)

    if not loop_dir.exists():
        raise FileNotFoundError(f"loop directory {loop_dir} not found")
    if not exts:
        raise ValueError("no extensions specified")

    from utilities.loop_scanner import scan_loops as scan_generic

    normalized = {e.replace("midi", "mid") for e in exts}
    loop_files = scan_generic(loop_dir, list(normalized))
    for p in loop_files:
        suf = p.suffix.lower().lstrip(".")
        suf = "mid" if suf in {"mid", "midi"} else suf
        if suf not in normalized:
            continue
        seq: list[State] = []
        if p.suffix.lower() in {".mid", ".midi"}:
            iterator = _iter_midi(p)
        else:
            iterator = _iter_wav(p)
        for step, lbl, vel, micro in iterator:
            seq.append(((step % RESOLUTION), lbl))
            vel_map[lbl].append(vel)
            micro_map[lbl].append(micro)
        if seq:
            events.append((sorted(seq, key=lambda x: x[0]), p.name))
    mean_velocity = {k: sum(v) / len(v) for k, v in vel_map.items() if v}
    vel_deltas = {
        k: Counter(int(v - mean_velocity[k]) for v in vals)
        for k, vals in vel_map.items()
    }
    micro_offsets = {k: Counter(vals) for k, vals in micro_map.items()}
    return events, mean_velocity, vel_deltas, micro_offsets
