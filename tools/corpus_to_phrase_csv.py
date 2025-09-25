from __future__ import annotations

"""Convert MIDI or corpus samples to phrase CSVs."""

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Pattern

# pretty_midi は MIDI 入力時のみ必要。
# from-corpus モードでは読み込まず、警告も出さないため遅延インポートにする。
try:  # pragma: no cover - PyYAML optional for --help
    import yaml
except Exception:  # pragma: no cover - no yaml
    yaml = None  # type: ignore

try:  # pragma: no cover - pretty_midi optional for labeling
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - fallback when pretty_midi missing
    pretty_midi = None  # type: ignore


INSTRUMENT_RANGES = {
    "bass": (28, 52),
    "piano": (21, 108),
    "strings": (40, 84),
    "guitar_low": (40, 64),
    "guitar_lead": (52, 88),
}

FIELDS = [
    "pitch",
    "velocity",
    "duration",
    "pos",
    "boundary",
    "bar",
    "instrument",
    "program",
    "song_id",
    "velocity_bucket",
    "duration_bucket",
]


def _program_name(pgm: object) -> str:
    try:
        value = int(pgm)  # type: ignore[arg-type]
    except Exception:
        return str(pgm)
    if pretty_midi is not None:
        try:
            return pretty_midi.program_to_instrument_name(value)
        except Exception:
            return str(value)
    return str(value)


def _program_sort_key(val: str) -> tuple[int, int | str]:
    try:
        return (0, int(val))
    except Exception:
        return (1, val)


def summarize_tags(rows: Iterable[dict[str, object]], vocab: dict[str, list[str]]) -> None:
    counts: dict[str, Counter[str]] = {k: Counter() for k in vocab}
    for row in rows:
        for key in vocab:
            if key in row:
                counts[key][str(row[key])] += 1
    for key, allowed in vocab.items():
        seen = counts[key]
        unknown = {v: c for v, c in seen.items() if v not in allowed}
        missing = [v for v in allowed if v not in seen]
        print(f"{key}: unknown={unknown} missing={missing}")


def bin_duration(duration_beats: float, bins: int) -> int:
    """Map a duration in beats to a discrete bin index."""

    return max(0, min(bins - 1, round(duration_beats * bins)))


def bin_velocity(velocity: int, bins: int) -> int:
    """Map velocity 0-127 to a discrete bin index."""

    return max(0, min(bins - 1, velocity * bins // 128))


def deterministic_split(
    files: list[Path], valid_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split files into train/valid using a stable deterministic order.

    Historically the tool preserved alphabetical order for small corpora.
    Tests and downstream tooling rely on that behaviour (e.g. expecting the
    first file lexicographically to land in the training split).  We therefore
    sort the input paths and avoid shuffling; callers wanting a randomised
    split can opt into ``--hash-split`` instead.  The ``seed`` argument is kept
    for backward compatibility and future extensions.
    """

    unused_seed = seed  # retained for signature compatibility / future use
    del unused_seed

    ordered = sorted(files)
    if not ordered:
        return [], []
    split = max(1, int(len(ordered) * (1 - valid_ratio)))
    return ordered[:split], ordered[split:]


def hash_split(files: list[Path], valid_ratio: float) -> tuple[list[Path], list[Path]]:
    import hashlib

    train: list[Path] = []
    valid: list[Path] = []
    for p in files:
        h = hashlib.md5(p.stem.encode()).hexdigest()
        frac = int(h, 16) / 16**32
        if frac < (1 - valid_ratio):
            train.append(p)
        else:
            valid.append(p)
    return train, valid


def midi_to_rows(
    path: Path,
    gap: float,
    sections: list[tuple[float, str]] | None,
    instrument_filter: str | None,
    instrument_regex: Pattern[str] | None,
    pitch_range: tuple[int, int] | None,
    path_hint_match: bool = False,
    *,
    emit_buckets: bool = False,
    dur_bins: int = 16,
    vel_bins: int = 8,
) -> tuple[list[dict[str, object]], bool]:
    """Parse *path* into phrase rows using simple boundary heuristics.

    Returns the parsed rows and whether any instrument matched *instrument_filter*.
    """
    # 遅延インポート: MIDI 入力時にのみ pretty_midi を読み込む
    try:  # pragma: no cover - optional dependency
        import pretty_midi  # type: ignore
    except Exception as e:  # pragma: no cover - not installed
        raise RuntimeError(
            "pretty_midi is required to parse MIDI files (pip install pretty_midi)"
        ) from e
    pm = pretty_midi.PrettyMIDI(str(path))
    ticks_per_beat = pm.resolution

    # Convert provided section-change times (seconds) into beats for consistent comparisons
    sections_beats: list[tuple[float, str]] | None = None
    if sections:
        try:
            sections_beats = [
                (pm.time_to_tick(float(t)) / float(ticks_per_beat), str(s)) for (t, s) in sections
            ]
        except Exception:
            # Fallback: if conversion fails, keep as-is (best-effort)
            sections_beats = [(float(t), str(s)) for (t, s) in sections]

    notes: list[tuple[float, float, int, int, str, int]] = []
    inst_match = False
    for inst in pm.instruments:
        name = inst.name or pretty_midi.program_to_instrument_name(inst.program)
        is_drum = bool(getattr(inst, "is_drum", False))
        try:
            prog_raw = int(getattr(inst, "program", -1))
        except Exception:
            prog_raw = -1
        prog_val = 128 if is_drum else (prog_raw if prog_raw >= 0 else -1)
        # Instrument selection logic:
        # 1) Prefer explicit instrument filters (substring/regex) against the MIDI track name.
        # 2) If those do not match but the file/folder path hints match the instrument, fall back to including.
        # 3) If no instrument filter is provided, include all instruments.
        selected = True
        has_any_filter = bool(instrument_filter or instrument_regex)
        if has_any_filter:
            name_lc = (name or "").lower()
            name_ok = bool(instrument_filter and instrument_filter in name_lc)
            regex_ok = bool(instrument_regex and instrument_regex.search(name))
            if name_ok or regex_ok:
                selected = True
                inst_match = True
            elif path_hint_match:
                # Path-based fallback: allow even if MIDI track name didn't match
                selected = True
                inst_match = True
            else:
                selected = False
        if not selected:
            continue
        for n in inst.notes:
            if pitch_range and not (pitch_range[0] <= n.pitch <= pitch_range[1]):
                continue
            start_tick = pm.time_to_tick(n.start)
            end_tick = pm.time_to_tick(n.end)
            start_beats = start_tick / ticks_per_beat
            dur_beats = (end_tick - start_tick) / ticks_per_beat
            notes.append((start_beats, dur_beats, n.pitch, n.velocity, name, prog_val))
    notes.sort()

    rows: list[dict[str, object]] = []
    prev_start = None
    prev_bar = None
    prev_section = None

    for start, dur, pitch, vel, name, prog_val in notes:
        bar = int(start // 4)
        pos = int((start % 4) * ticks_per_beat)
        boundary = 0
        cur_section = None
        if sections_beats:
            for t, s in sections_beats:
                if start >= t:
                    cur_section = s
                else:
                    break
        if (
            prev_start is None
            or bar != prev_bar
            or start - prev_start >= gap
            or cur_section != prev_section
        ):
            boundary = 1
        row = {
            "pitch": pitch,
            "velocity": vel,
            "duration": dur,
            "pos": pos,
            "boundary": boundary,
            "bar": bar,
            "instrument": name,
            "program": prog_val,
            "velocity_bucket": -1,
            "duration_bucket": -1,
        }
        if emit_buckets:
            row["velocity_bucket"] = bin_velocity(vel, vel_bins)
            row["duration_bucket"] = bin_duration(dur, dur_bins)
        rows.append(row)
        prev_start = start
        prev_bar = bar
        prev_section = cur_section
    return rows, inst_match


def write_csv(rows: Iterable[dict[str, object]], path: Path, fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_samples_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def list_instruments(
    root: Path,
    *,
    min_count: int = 1,
    stats_json: Path | None = None,
    examples_per_key: int = 0,
    json_out: bool = False,
) -> bool:
    """Print instrument statistics and return the total keys across categories.

    The return value is the sum of unique keys meeting ``min_count`` across all
    tracked categories (instrument, track_name, program, path). This matches the
    CLI's legacy behavior of reporting the combined total rather than a single
    category count.
    """

    counters = {
        "instrument": Counter(),
        "track_name": Counter(),
        "program": Counter(),
        "path": Counter(),
    }
    examples = {k: defaultdict(list) for k in counters}
    for split in ("train", "valid"):
        base = root / split
        files: list[Path] = []
        if (base / "samples.jsonl").is_file():
            files = [base / "samples.jsonl"]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        # Fallback to flat JSONL layout (e.g., root/train.jsonl)
        alt_file = root / f"{split}.jsonl"
        if not files and alt_file.is_file():
            files = [alt_file]
        sample_counter = 0
        sample_counter = 0
        for p in files:
            for obj in read_samples_jsonl(p):
                meta = obj.get("meta", {})
                fname = str(
                    obj.get("path")
                    or obj.get("source_path")
                    or obj.get("filename")
                    or obj.get("file")
                    or meta.get("path")
                    or meta.get("source_path")
                    or meta.get("filename")
                    or meta.get("file")
                    or ""
                )
                stem = Path(fname).stem.lower()
                inst = obj.get("instrument")
                if inst is None:
                    inst = meta.get("instrument")
                if inst:
                    key = str(inst).lower()
                    counters["instrument"][key] += 1
                    if stem and len(examples["instrument"][key]) < examples_per_key:
                        examples["instrument"][key].append(stem)
                track = obj.get("track_name")
                if track is None:
                    track = meta.get("track_name")
                if track:
                    key = str(track).lower()
                    counters["track_name"][key] += 1
                    if stem and len(examples["track_name"][key]) < examples_per_key:
                        examples["track_name"][key].append(stem)
                prog = obj.get("program")
                if prog is None and "program" in meta:
                    prog = meta["program"]
                if prog is not None:
                    key_prog = int(prog)
                    counters["program"][key_prog] += 1
                    if stem and len(examples["program"][key_prog]) < examples_per_key:
                        examples["program"][key_prog].append(stem)
                if stem:
                    counters["path"][stem] += 1
                    if len(examples["path"][stem]) < examples_per_key:
                        examples["path"][stem].append(stem)
    counts: dict[str, dict[object, int]] = {}
    for key, ctr in counters.items():
        counts[key] = {val: ctr[val] for val in ctr if ctr[val] >= min_count}
    if stats_json:
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(json.dumps(counts, indent=2))
    if json_out:
        payload: dict[str, dict[str, dict[str, object]]] = {}
        for key, ctr in counters.items():
            entries: dict[str, dict[str, object]] = {}
            for val, count in ctr.items():
                if count < min_count:
                    continue
                entry_examples = list(examples[key].get(val, []))
                entries[str(val)] = {"count": count, "examples": entry_examples}
            payload[key] = entries
        programs = payload.get("program", {})
        if programs:
            instrument_meta: list[dict[str, object]] = []
            for pgm in sorted(programs.keys(), key=_program_sort_key):
                try:
                    pgm_val: object = int(pgm)
                except Exception:
                    pgm_val = pgm
                instrument_meta.append({"program": pgm_val, "name": _program_name(pgm)})
            payload["instruments"] = instrument_meta
        print(json.dumps(payload, indent=2))
    else:
        for key, ctr in counters.items():
            print(f"{key}:")
            for val, count in ctr.most_common():
                if count < min_count:
                    continue
                ex = examples[key].get(val)
                if ex:
                    print(f"  {val}: {count} (examples: {', '.join(ex)})")
                else:
                    print(f"  {val}: {count}")
    return sum(len(vals) for vals in counts.values())


def corpus_mode(
    root: Path,
    *,
    emit_buckets: bool = False,
    dur_bins: int = 16,
    vel_bins: int = 8,
    instrument: str | None = None,
    instrument_regex: Pattern[str] | None = None,
    pitch_range: tuple[int, int] | None = None,
    include_programs: set[int] | None = None,
    strict_all: bool = False,
    min_notes_per_sample: int = 0,
    token_ioi_gap: float = 0.0,
    token_pitch_leap: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    instrument = instrument.lower() if instrument else None

    stats_all: dict[str, dict[str, object]] = {}

    def load_split(split: str) -> list[dict[str, object]]:
        base = root / split
        files: list[Path] = []
        jsonl_file = base / "samples.jsonl"
        samples_glob = base / "samples/*.jsonl"
        alt_file = root / f"{split}.jsonl"
        if jsonl_file.is_file():
            files = [jsonl_file]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        if not files and alt_file.is_file():
            files = [alt_file]
        if not files:
            available = sorted(str(p) for p in root.parent.glob("*") if p.is_dir())
            if available:
                candidates = "\n".join(f"  - {p}" for p in available)
            else:
                candidates = "  (no corpus directories found)"
            msg = (
                "Corpus is empty or missing: "
                f"checked {jsonl_file}, {samples_glob}, and {alt_file} (from --from-corpus {root}).\n"
                "--from-corpus may be incorrect. Available corpus directories:\n"
                f"{candidates}\n"
                "Example layout:\n"
                "  data/corpus/NAME/train/samples.jsonl"
            )
            print(msg, file=sys.stderr)
            sys.exit(1)
        rows: list[dict[str, object]] = []
        stats = Counter()
        pitch_hist: Counter[int] = Counter()
        vel_hist: Counter[int] = Counter()
        dur_hist: Counter[int] = Counter()
        inst_hist: Counter[str] = Counter()
        track_hist: Counter[str] = Counter()
        prog_hist: Counter[int] = Counter()
        sample_counter = 0
        for p in files:
            for obj in read_samples_jsonl(p):
                meta = obj.get("meta", {})
                name = str(obj.get("instrument") or meta.get("instrument") or "")
                track = str(obj.get("track_name") or meta.get("track_name") or "")
                program = obj.get("program")
                if program is None and "program" in meta:
                    program = meta["program"]
                prog_int = int(program) if program is not None else None
                is_drum = bool(obj.get("is_drum") or meta.get("is_drum"))
                prog_value = 128 if is_drum else (prog_int if prog_int is not None else -1)
                # Support two corpus formats:
                # 1) "samples" JSONL with explicit per-note fields (pitch/velocity/duration/...)
                # 2) Token corpora with NOTE_x and DUV_d_v (or D_/V_) sequences
                if "tokens" in obj and isinstance(obj["tokens"], list):
                    tokens = [str(t) for t in obj["tokens"]]
                    # Derive beats-per-bar if present (default to 4)
                    try:
                        beats_per_bar = float(meta.get("beats_per_bar", 4.0))
                    except Exception:
                        beats_per_bar = 4.0

                    # Counters for min-notes filter and position tracking within the sequence
                    note_tokens = [t for t in tokens if t.startswith("NOTE_")]
                    note_count = len(note_tokens)
                    if note_count < min_notes_per_sample:
                        stats["removed_by_min_notes"] += 1
                        stats["removed"] += 1
                        continue

                    # Optional sample-level filtering for instrument hints via source_path/track/name
                    # Many token corpora omit explicit instrument fields; fall back to path stem.
                    keep_sample = True
                    if instrument or instrument_regex:
                        fname = str(
                            meta.get("source_path")
                            or meta.get("path")
                            or obj.get("path")
                            or obj.get("source_path")
                            or ""
                        )
                        stem = Path(fname).stem.lower() if fname else ""
                        nm = (name or "").lower()
                        tr = (track or "").lower()
                        name_ok = bool(instrument and (instrument in nm or instrument in tr or instrument in stem))
                        regex_ok = bool(
                            instrument_regex
                            and (
                                instrument_regex.search(nm)
                                or instrument_regex.search(tr)
                                or instrument_regex.search(stem)
                            )
                        )
                        if strict_all:
                            keep_sample = (not instrument or name_ok) and (not instrument_regex or regex_ok)
                            if not keep_sample:
                                if instrument and not name_ok:
                                    stats["removed_by_name"] += 1
                                if instrument_regex and not regex_ok:
                                    stats["removed_by_regex"] += 1
                                stats["removed"] += 1
                                continue
                        else:
                            keep_sample = name_ok or regex_ok
                            if (instrument or instrument_regex) and not keep_sample:
                                if instrument and not name_ok:
                                    stats["removed_by_name"] += 1
                                if instrument_regex and not regex_ok:
                                    stats["removed_by_regex"] += 1
                                stats["removed"] += 1
                                continue

                    cur_time_beats = 0.0
                    prev_local_bar: int | None = None
                    cur_bar = 0  # global (with offset) tracker for pos reset
                    pos_idx_in_bar = 0
                    bar_offset = sample_counter * 10000
                    prev_pitch: int | None = None
                    prev_dur_beats: float = 0.0
                    # Walk NOTE/(DUV|D/V) pairs in order
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i]
                        if not tok.startswith("NOTE_"):
                            i += 1
                            continue
                        try:
                            pitch = int(tok.split("_", 1)[1])
                        except Exception:
                            i += 1
                            continue
                        dur_beats: float = 0.0
                        vel_val: int = 0
                        vb_idx: int | None = None
                        db_idx: int | None = None
                        if i + 1 < len(tokens):
                            nxt = tokens[i + 1]
                            if nxt.startswith("DUV_"):
                                parts = nxt.split("_")
                                if len(parts) == 3:
                                    try:
                                        db_idx = int(parts[1])
                                        vb_idx = int(parts[2])
                                    except Exception:
                                        db_idx = vb_idx = None
                            elif nxt.startswith("D_") or nxt.startswith("V_"):
                                # Handle non-combined duration/velocity tokens
                                j = i + 1
                                while j < len(tokens) and (tokens[j].startswith("D_") or tokens[j].startswith("V_")):
                                    if tokens[j].startswith("D_"):
                                        try:
                                            db_idx = int(tokens[j].split("_", 1)[1])
                                        except Exception:
                                            pass
                                    elif tokens[j].startswith("V_"):
                                        try:
                                            vb_idx = int(tokens[j].split("_", 1)[1])
                                        except Exception:
                                            pass
                                    j += 1
                        # Approximate continuous values from bucket indices when available
                        if db_idx is not None:
                            try:
                                dur_beats = max(0.0, float(db_idx) / max(1, dur_bins))
                            except Exception:
                                dur_beats = 0.0
                        if vb_idx is not None:
                            try:
                                # Map velocity bucket index back to 0..127 via bin midpoint
                                step = 128 / max(1, vel_bins)
                                vel_val = int(min(127, max(0, round((vb_idx + 0.5) * step))))
                            except Exception:
                                vel_val = 0

                        # Compute bar/pos and boundary using beat position
                        start_beats = cur_time_beats
                        local_bar_idx = int(start_beats // max(1.0, beats_per_bar))
                        bar_idx = local_bar_idx + bar_offset
                        # Use ordinal position within the (local) bar to keep values small and ordered
                        if prev_local_bar is None or local_bar_idx != prev_local_bar:
                            cur_bar = bar_idx
                            pos_idx_in_bar = 0
                        pos = int(pos_idx_in_bar)
                        boundary_flag = 1 if (prev_local_bar is None or local_bar_idx != prev_local_bar) else 0
                        # Optional token-heuristics: large preceding IOI or large pitch leap
                        if not boundary_flag:
                            if token_ioi_gap and token_ioi_gap > 0.0:
                                if prev_dur_beats >= float(token_ioi_gap):
                                    boundary_flag = 1
                            if not boundary_flag and token_pitch_leap and token_pitch_leap > 0 and prev_pitch is not None:
                                try:
                                    if abs(int(pitch) - int(prev_pitch)) >= int(token_pitch_leap):
                                        boundary_flag = 1
                                except Exception:
                                    pass

                        # Apply optional filters
                        keep_note = True
                        if pitch_range:
                            keep_note = pitch_range[0] <= pitch <= pitch_range[1]
                            if not keep_note:
                                stats["removed_by_pitch"] += 1
                                stats["removed"] += 1
                        if keep_note:
                            row = {
                                "pitch": pitch,
                                "velocity": vel_val,
                                "duration": dur_beats,
                                "pos": pos,
                                "boundary": boundary_flag,
                                "bar": bar_idx,
                                "instrument": name,
                                "program": prog_value,
                                "velocity_bucket": -1,
                                "duration_bucket": -1,
                            }
                            if emit_buckets:
                                if vb_idx is not None:
                                    row["velocity_bucket"] = int(vb_idx)
                                else:
                                    row["velocity_bucket"] = bin_velocity(int(row["velocity"]), vel_bins)
                                if db_idx is not None:
                                    row["duration_bucket"] = int(db_idx)
                                else:
                                    row["duration_bucket"] = bin_duration(float(row["duration"]), dur_bins)
                            rows.append(row)
                            stats["kept"] += 1
                            pitch_hist[pitch] += 1
                            vel_hist[int(row["velocity"])] += 1
                            dur_hist[int(round(float(row["duration"]) * 1000))] += 1

                        # Advance cursors
                        prev_local_bar = local_bar_idx
                        prev_pitch = int(pitch)
                        prev_dur_beats = float(max(0.0, dur_beats))
                        pos_idx_in_bar += 1
                        cur_time_beats += max(0.0, float(dur_beats))
                        i += 2  # Skip NOTE and its following (DUV|D|V)
                        continue
                    # Finished token-sequence sample
                    # Histograms for inst/track/program gathered above
                    sample_counter += 1
                    continue

                # Legacy/explicit-sample format path (expects pitch field)
                pitch = int(obj["pitch"])
                fname = str(
                    obj.get("path")
                    or obj.get("source_path")
                    or obj.get("filename")
                    or obj.get("file")
                    or meta.get("path")
                    or meta.get("source_path")
                    or meta.get("filename")
                    or meta.get("file")
                    or ""
                )
                missing_fields = []
                if not name:
                    missing_fields.append("instrument")
                if not track:
                    missing_fields.append("track_name")
                if prog_int is None:
                    missing_fields.append("program")
                if not fname:
                    missing_fields.append("path")
                if missing_fields:
                    logging.warning("Sample missing %s in %s", ", ".join(missing_fields), p)
                stem = Path(fname).stem.lower()
                inst_hist[name.lower()] += 1
                track_hist[track.lower()] += 1
                if prog_int is not None:
                    prog_hist[prog_int] += 1
                elif is_drum:
                    prog_hist[128] += 1
                note_count = int(
                    obj.get("note_count")
                    or (len(obj.get("notes", [])) if isinstance(obj.get("notes"), list) else 1)
                )
                if note_count < min_notes_per_sample:
                    stats["removed_by_min_notes"] += 1
                    stats["removed"] += 1
                    continue
                name_ok = False
                regex_ok = False
                pitch_ok = False
                program_ok = False
                if instrument:
                    nm = name.lower()
                    if instrument in nm or instrument in track.lower() or instrument in stem:
                        name_ok = True
                if instrument_regex:
                    if (
                        instrument_regex.search(name)
                        or instrument_regex.search(track)
                        or instrument_regex.search(stem)
                    ):
                        regex_ok = True
                if pitch_range:
                    pitch_ok = pitch_range[0] <= pitch <= pitch_range[1]
                if include_programs is not None and prog_int is not None:
                    program_ok = prog_int in include_programs
                keep = True
                if strict_all:
                    if instrument and not name_ok:
                        stats["removed_by_name"] += 1
                        keep = False
                    if instrument_regex and not regex_ok:
                        stats["removed_by_regex"] += 1
                        keep = False
                    if pitch_range and not pitch_ok:
                        stats["removed_by_pitch"] += 1
                        keep = False
                    if include_programs and not program_ok:
                        stats["removed_by_program"] += 1
                        keep = False
                else:
                    if instrument or instrument_regex or pitch_range or include_programs:
                        keep = name_ok or regex_ok or pitch_ok or program_ok
                        if not keep:
                            if instrument and not name_ok:
                                stats["removed_by_name"] += 1
                            if instrument_regex and not regex_ok:
                                stats["removed_by_regex"] += 1
                            if pitch_range and not pitch_ok:
                                stats["removed_by_pitch"] += 1
                            if include_programs and not program_ok:
                                stats["removed_by_program"] += 1
                if not keep:
                    stats["removed"] += 1
                    continue
                row = {
                    "pitch": pitch,
                    "velocity": obj.get("velocity", 0),
                    "duration": obj.get("duration", 0),
                    "pos": obj.get("pos", 0),
                    "boundary": obj.get("boundary", 0),
                    "bar": obj.get("bar", 0),
                    "instrument": name,
                    "program": prog_value,
                    "velocity_bucket": -1,
                    "duration_bucket": -1,
                }
                if emit_buckets:
                    row["velocity_bucket"] = bin_velocity(int(row["velocity"]), vel_bins)
                    row["duration_bucket"] = bin_duration(float(row["duration"]), dur_bins)
                rows.append(row)
                stats["kept"] += 1
                pitch_hist[pitch] += 1
                vel_hist[int(row["velocity"])] += 1
                dur_hist[int(round(float(row["duration"]) * 1000))] += 1
        stats_dict = {
            "kept": stats["kept"],
            "removed": stats["removed"],
            "removed_by_name": stats["removed_by_name"],
            "removed_by_regex": stats["removed_by_regex"],
            "removed_by_pitch": stats["removed_by_pitch"],
            "removed_by_program": stats["removed_by_program"],
            "removed_by_min_notes": stats["removed_by_min_notes"],
            "pitch_hist": dict(pitch_hist),
            "velocity_hist": dict(vel_hist),
            "duration_hist": {str(k / 1000): v for k, v in dur_hist.items()},
            "instrument_hist": dict(inst_hist),
            "track_name_hist": dict(track_hist),
            "program_hist": dict(prog_hist),
        }
        stats_all[split] = stats_dict
        logging.info("%s stats %s", split, stats_dict)
        return rows

    train_rows = load_split("train")
    valid_rows = load_split("valid")
    return train_rows, valid_rows, stats_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path)
    parser.add_argument("--out-train", type=Path)
    parser.add_argument("--out-valid", type=Path)
    parser.add_argument(
        "--instrument",
        help="case-insensitive substring filter (OR with --pitch-range in corpus mode)",
    )
    parser.add_argument(
        "--instrument-regex",
        help="regex instrument filter (OR with --pitch-range in corpus mode)",
    )
    parser.add_argument("--boundary-gap-beats", type=float, default=0.5)
    parser.add_argument("--boundary-on-section-change", action="store_true")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-notes", type=int, default=0)
    parser.add_argument(
        "--pitch-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="inclusive pitch filter; typical bass range 28-60",
    )
    parser.add_argument("--instrument-name", choices=sorted(INSTRUMENT_RANGES))
    parser.add_argument("--max-bars", type=int, default=10**9)
    parser.add_argument(
        "--hash-split",
        action="store_true",
        help="deterministic split by file hash instead of RNG shuffle",
    )
    parser.add_argument("--from-corpus", dest="from_corpus", type=Path)
    parser.add_argument(
        "--emit-buckets",
        action="store_true",
        help="compute velocity/duration buckets (columns always present, -1 when disabled)",
    )
    parser.add_argument("--dur-bins", type=int, default=16)
    parser.add_argument("--vel-bins", type=int, default=8)
    parser.add_argument("--tag-vocab-in", type=Path)
    parser.add_argument("--tag-vocab-out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate without writing CSV")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="like --dry-run plus tag vocab check",
    )
    parser.add_argument("--list-instruments", action="store_true")
    parser.add_argument("--strict-all", action="store_true")
    parser.add_argument("--min-notes-per-sample", type=int, default=0)
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--examples-per-key", type=int, default=0)
    parser.add_argument(
        "--include-programs",
        type=int,
        nargs="+",
        metavar="PROG",
        help="keep only samples with these GM program numbers",
    )
    parser.add_argument("--duv-mode", choices=["reg", "cls", "both"], default="reg")
    parser.add_argument(
        "--no-path-hint",
        action="store_true",
        help="disable path-based instrument hinting when filtering in --in (MIDI) mode",
    )
    # token-corpus heuristic options (disabled by default)
    parser.add_argument(
        "--token-ioi-gap",
        type=float,
        default=0.0,
        help="in token corpora, mark boundary when preceding note duration (approx IOI) >= this many beats",
    )
    parser.add_argument(
        "--token-pitch-leap",
        type=int,
        default=0,
        help="in token corpora, mark boundary when abs pitch leap >= this many semitones",
    )
    args = parser.parse_args(argv)

    inst_re = re.compile(args.instrument_regex, re.I) if args.instrument_regex else None

    pitch_range: tuple[int, int] | None = None
    if args.instrument_name:
        pitch_range = INSTRUMENT_RANGES[args.instrument_name]
    if args.pitch_range:
        pitch_range = tuple(args.pitch_range)

    include_programs: set[int] | None = None
    if args.include_programs:
        include_programs = set(args.include_programs)

    if args.validate_only:
        args.dry_run = True

    if args.duv_mode in {"cls", "both"} and not args.emit_buckets:
        logging.warning("duv_mode %s requires buckets; enabling --emit-buckets", args.duv_mode)
        args.emit_buckets = True

    if not args.list_instruments and (args.out_train is None or args.out_valid is None):
        parser.error("--out-train and --out-valid are required unless --list-instruments")

    if args.list_instruments:
        if not args.from_corpus:
            raise SystemExit("--list-instruments requires --from-corpus")
        ok = list_instruments(
            args.from_corpus,
            min_count=args.min_count,
            stats_json=args.stats_json,
            examples_per_key=args.examples_per_key,
            json_out=args.json,
        )
        return 0 if ok else 1

    if args.from_corpus:
        train_rows, valid_rows, stats = corpus_mode(
            args.from_corpus,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
            instrument=args.instrument,
            instrument_regex=inst_re,
            pitch_range=pitch_range,
            include_programs=include_programs,
            strict_all=args.strict_all,
            min_notes_per_sample=args.min_notes_per_sample,
            token_ioi_gap=args.token_ioi_gap,
            token_pitch_leap=args.token_pitch_leap,
        )
        if args.stats_json:
            args.stats_json.parent.mkdir(parents=True, exist_ok=True)
            args.stats_json.write_text(json.dumps(stats, indent=2))
        if not train_rows or not valid_rows:
            for split, rows in (("train", train_rows), ("valid", valid_rows)):
                if not rows:
                    st = stats.get(split, {})
                    print(f"{split} histograms:")
                    print(f"  instruments: {st.get('instrument_hist', {})}")
                    print(f"  track_names: {st.get('track_name_hist', {})}")
                    print(f"  programs: {st.get('program_hist', {})}")
            print(
                "No rows matched filters. Hints: try --pitch-range 28 60 (bass), "
                "relax --instrument/--instrument-regex, run --list-instruments."
            )
            return 1
        if args.validate_only and args.tag_vocab_in:
            vocab = json.loads(args.tag_vocab_in.read_text())
            summarize_tags(train_rows + valid_rows, vocab)
        if args.dry_run or args.validate_only:
            for split, st in stats.items():
                logging.info(
                    "%s kept %d removed %d name_miss=%d regex_miss=%d pitch_miss=%d min_notes=%d",
                    split,
                    st.get("kept", 0),
                    st.get("removed", 0),
                    st.get("removed_by_name", 0),
                    st.get("removed_by_regex", 0),
                    st.get("removed_by_pitch", 0),
                    st.get("removed_by_min_notes", 0),
                )
            return 0 if train_rows and valid_rows else 1
        write_csv(train_rows, args.out_train)
        write_csv(valid_rows, args.out_valid)
        if args.tag_vocab_in and args.tag_vocab_out:
            if not args.tag_vocab_in.is_file():
                raise SystemExit(f"tag vocab not found: {args.tag_vocab_in}")
            args.tag_vocab_out.parent.mkdir(parents=True, exist_ok=True)
            args.tag_vocab_out.write_text(args.tag_vocab_in.read_text())
        return 0

    if args.src is None:
        raise SystemExit("--in is required when not using --from-corpus")

    files = [p for p in args.src.rglob("*") if p.suffix.lower() in {".mid", ".midi"}]
    if not files:
        raise SystemExit(
            f"No MIDI files found under {args.src}. Place .mid/.midi files (Unicode/日本語のパス対応) and retry."
        )

    tags: dict[str, list[tuple[float, str]]] = {}
    if args.boundary_on_section_change:
        tag_path = args.src / "tags.yaml"
        if tag_path.is_file():
            if yaml is None:
                raise RuntimeError("PyYAML required to read tags.yaml (pip install PyYAML)")
            raw = yaml.safe_load(tag_path.read_text()) or {}
            for fname, info in raw.items():
                sections = info.get("section") or []
                tags[fname] = [(float(t), str(s)) for t, s in sections]

    if args.hash_split:
        train_files, valid_files = hash_split(files, args.valid_ratio)
    else:
        train_files, valid_files = deterministic_split(files, args.valid_ratio, args.seed)

    train_rows: list[dict[str, object]] = []
    train_stats = Counter()
    for p in train_files:
        sec = tags.get(p.name)
        # Path-based instrument hint: allow folder/file names to drive selection when MIDI names are unhelpful.
        path_str = "/".join([part for part in p.parts]).lower()
        path_hint = False
        if not args.no_path_hint:
            if args.instrument:
                try:
                    if args.instrument.lower() in path_str:
                        path_hint = True
                except Exception:
                    pass
            if inst_re and not path_hint:
                try:
                    if inst_re.search(path_str):
                        path_hint = True
                except Exception:
                    pass

        rows, matched = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
            inst_re,
            pitch_range,
            path_hint_match=path_hint,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
        )
        # attach song_id from filename stem
        sid = p.stem
        for r in rows:
            r["song_id"] = sid
        if not rows:
            if args.instrument and not matched:
                train_stats["instrument"] += 1
            else:
                train_stats["empty"] += 1
            continue
        if len(rows) < args.min_notes:
            train_stats["min_notes"] += 1
            continue
        if (rows[-1]["bar"] + 1) > args.max_bars:
            train_stats["max_bars"] += 1
            continue
        train_rows.extend(rows)
        train_stats["kept"] += 1

    valid_rows: list[dict[str, object]] = []
    valid_stats = Counter()
    for p in valid_files:
        sec = tags.get(p.name)
        path_str = "/".join([part for part in p.parts]).lower()
        path_hint = False
        if args.instrument:
            try:
                if args.instrument.lower() in path_str:
                    path_hint = True
            except Exception:
                pass
        if inst_re and not path_hint:
            try:
                if inst_re.search(path_str):
                    path_hint = True
            except Exception:
                pass

        rows, matched = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
            inst_re,
            pitch_range,
            path_hint_match=path_hint,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
        )
        sid = p.stem
        for r in rows:
            r["song_id"] = sid
        if not rows:
            if args.instrument and not matched:
                valid_stats["instrument"] += 1
            else:
                valid_stats["empty"] += 1
            continue
        if len(rows) < args.min_notes:
            valid_stats["min_notes"] += 1
            continue
        if (rows[-1]["bar"] + 1) > args.max_bars:
            valid_stats["max_bars"] += 1
            continue
        valid_rows.extend(rows)
        valid_stats["kept"] += 1

    logging.info(
        "train stats %s | valid stats %s",
        dict(train_stats),
        dict(valid_stats),
    )

    if not train_rows or not valid_rows:
        raise SystemExit(
            f"no rows after filtering: train={dict(train_stats)} valid={dict(valid_stats)}"
        )

    if args.validate_only and args.tag_vocab_in:
        vocab = json.loads(args.tag_vocab_in.read_text())
        summarize_tags(train_rows + valid_rows, vocab)

    if args.dry_run or args.validate_only:
        logging.info("train rows %d valid rows %d", len(train_rows), len(valid_rows))
    else:
        write_csv(train_rows, args.out_train)
        write_csv(valid_rows, args.out_valid)
        if args.tag_vocab_in and args.tag_vocab_out:
            if not args.tag_vocab_in.is_file():
                raise SystemExit(f"tag vocab not found: {args.tag_vocab_in}")
            args.tag_vocab_out.parent.mkdir(parents=True, exist_ok=True)
            args.tag_vocab_out.write_text(args.tag_vocab_in.read_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
