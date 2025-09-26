from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None  # type: ignore
try:
    from . import pb_math  # package-relative; adjust sys.path if run as script
except ImportError:  # pragma: no cover - fallback for direct path execution
    from utilities import pb_math


@dataclass
class NoteRow:
    pitch: int
    duration: float
    bar: int
    position: int
    velocity: int
    program: int
    chord_symbol: str
    articulation: str
    q_onset: float
    q_duration: float
    cc64: Optional[int] = None
    cc64_ratio: Optional[float] = None
    cc11_at_onset: Optional[int] = None
    cc11_mean: Optional[float] = None
    bend: Optional[int] = None
    bend_range: Optional[float] = None
    bend_max_semi: Optional[float] = None
    bend_rms_semi: Optional[float] = None
    vib_rate_hz: Optional[float] = None


def _last_value(times: List[float], values: List[float], t: float) -> float:
    """Return the last value at or before ``t`` using binary search."""
    idx = bisect_right(times, t) - 1
    return values[idx] if idx >= 0 else 0


def _format_bend_range(val: Optional[float]) -> str:
    if val is None:
        return ""
    rounded = round(val)
    if abs(val - rounded) < 1e-6:
        return str(int(rounded))
    return f"{val:.6g}"


def _collect_bend_ranges(
    control_changes: List["pretty_midi.ControlChange"],
) -> Tuple[List[float], List[float]]:
    """Return times and values for pitch-bend range changes via RPN messages."""
    times: List[float] = [0.0]
    values: List[float] = [2.0]
    rpn_msb: Optional[int] = None
    rpn_lsb: Optional[int] = None
    pending_msb: Optional[int] = None
    def _sort_key(cc: "pretty_midi.ControlChange") -> tuple[float, int, int]:
        order = {101: 0, 100: 1, 6: 2, 38: 3}.get(cc.number, cc.number)
        return (float(cc.time), order, int(cc.number))

    for cc in sorted(control_changes, key=_sort_key):
        num = cc.number
        val = cc.value
        t = float(cc.time)
        if num == 101:  # RPN MSB
            rpn_msb = val
        elif num == 100:  # RPN LSB
            rpn_lsb = val
        elif num in (99, 98):  # NRPN select clears RPN
            rpn_msb = None
            rpn_lsb = None
            pending_msb = None
        elif num == 6:  # Data Entry MSB
            if rpn_msb == 0 and rpn_lsb == 0:
                times.append(t)
                values.append(float(val))
                pending_msb = val
        elif num == 38:  # Data Entry LSB
            if rpn_msb == 0 and rpn_lsb == 0 and pending_msb is not None:
                combined = float(pending_msb) + float(val) / 128.0
                values[-1] = combined
                pending_msb = None
    if pending_msb is not None and values:
        values[-1] = float(pending_msb)
    if len(values) > 1:
        first_real = next((v for v in values[1:] if abs(v - values[0]) > 1e-6), values[1])
        values[0] = first_real
    return times, values


def _active_ratio(
    times: List[float], values: List[int], start: float, end: float
) -> Optional[float]:
    """Return ratio of time CC value >=64 within [start, end]."""
    if not times or end <= start:
        return None
    duration = end - start
    events = [start] + [t for t in times if start < t < end] + [end]
    prev_val = _last_value(times, values, start)
    active = 0.0
    prev_time = start
    for t in events[1:]:
        if prev_val >= 64:
            active += t - prev_time
        prev_val = _last_value(times, values, t)
        prev_time = t
    return active / duration if duration > 0 else None


def _mean_value(
    times: List[float], values: List[int], start: float, end: float
) -> Optional[float]:
    """Return mean controller value within [start, end]."""
    if not times or end <= start:
        return None
    events = [start] + [t for t in times if start < t < end] + [end]
    prev_val = _last_value(times, values, start)
    total = 0.0
    prev_time = start
    for t in events[1:]:
        total += prev_val * (t - prev_time)
        prev_val = _last_value(times, values, t)
        prev_time = t
    duration = end - start
    return total / duration if duration > 0 else None


def _sample_bend(
    times: List[float], values: List[int], start: float, end: float, fps: int = 100
) -> List[float]:
    """Sample pitch-bend values linearly within [start, end]."""
    if not times or end <= start:
        return []

    # deduplicate consecutive identical pairs
    uniq_t: List[float] = []
    uniq_v: List[int] = []
    prev_t: Optional[float] = None
    prev_v: Optional[int] = None
    for t, v in zip(times, values):
        if prev_t is not None and t == prev_t and v == prev_v:
            continue
        uniq_t.append(t)
        uniq_v.append(v)
        prev_t, prev_v = t, v
    times, values = uniq_t, uniq_v

    duration = end - start
    max_points = 200
    n = max(int(duration * fps), 1)
    if n + 1 > max_points:
        n = max_points - 1
        step = duration / n
    else:
        step = 1.0 / fps

    samples: List[float] = []
    for i in range(n + 1):
        t = start + i * step
        idx = bisect_right(times, t) - 1
        if idx < 0:
            val = values[0]
        elif idx >= len(times) - 1:
            val = values[-1]
        else:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = values[idx], values[idx + 1]
            if t1 == t0:
                val = v1
            else:
                frac = (t - t0) / (t1 - t0)
                val = v0 + (v1 - v0) * frac
        samples.append(val)
    return samples


def _vibrato_rate(
    samples: List[float], duration: float, bend_range: int
) -> Optional[float]:
    if not samples or duration <= 0 or bend_range <= 0:
        return None
    semis = pb_math.pb_to_semi(samples, bend_range)
    signs = [1 if s > 0 else -1 if s < 0 else 0 for s in semis]
    crossings = 0
    for a, b in zip(signs, signs[1:]):
        if a != 0 and b != 0 and (a > 0) != (b > 0):
            crossings += 1
    if crossings < 2:
        return None
    return crossings / 2 / duration


def scan_midi_files(
    paths: Iterable[Path],
    include_cc: bool,
    include_bend: bool,
    bend_events: Optional[Path] = None,
    *,
    include_cc11: bool = False,
    cc_events: Optional[Path] = None,
) -> List[NoteRow]:
    """Return rows of rich note data from ``paths``."""
    if pretty_midi is None:  # pragma: no cover - handled at runtime
        raise RuntimeError("pretty_midi required")

    rows: List[NoteRow] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue

        if pm.time_signature_changes:
            ts = pm.time_signature_changes[0]
            beats_per_bar = ts.numerator * 4 / ts.denominator
        else:
            beats_per_bar = 4
        # TODO: support mid-piece time signature changes; only the first is used.
        subdiv = int(beats_per_bar * 4)

        def process(event_writer=None):
            for track_idx, inst in enumerate(pm.instruments):
                is_drum = bool(getattr(inst, "is_drum", False))
                try:
                    prog_raw = int(getattr(inst, "program", -1))
                except Exception:
                    prog_raw = -1
                prog_val = 128 if is_drum else (prog_raw if prog_raw >= 0 else -1)
                cc_times: List[float] = []
                cc_vals: List[int] = []
                cc11_times: List[float] = []
                cc11_vals: List[int] = []
                if include_cc:
                    pairs = sorted(
                        (float(cc.time), cc.value)
                        for cc in inst.control_changes
                        if cc.number == 64
                    )
                    if pairs:
                        cc_times, cc_vals = map(list, zip(*pairs))
                if include_cc11:
                    pairs = sorted(
                        (float(cc.time), cc.value)
                        for cc in inst.control_changes
                        if cc.number == 11
                    )
                    if pairs:
                        cc11_times, cc11_vals = map(list, zip(*pairs))
                bend_times: List[float] = []
                bend_vals: List[int] = []
                range_times: List[float] = []
                range_vals: List[int] = []
                if include_bend:
                    pairs = sorted((float(b.time), b.pitch) for b in inst.pitch_bends)
                    if pairs:
                        bend_times, bend_vals = map(list, zip(*pairs))
                    range_times, range_vals = _collect_bend_ranges(inst.control_changes)
                    if event_writer is not None:
                        for b in inst.pitch_bends:
                            t = float(b.time)
                            br = _last_value(range_times, range_vals, t)
                            semi = pb_math.pb_to_semi(b.pitch, br)
                            event_writer.writerow([t, b.pitch, semi, track_idx])

                for note in inst.notes:
                    start = float(note.start)
                    end = float(note.end)
                    tick = pm.time_to_tick(note.start)
                    end_tick = pm.time_to_tick(note.end)
                    q_onset = tick / pm.resolution
                    q_duration = (end_tick - tick) / pm.resolution
                    sixteenth = int(tick * 4 // pm.resolution)
                    bar = sixteenth // subdiv
                    position = sixteenth % subdiv
                    cc64 = (
                        int(round(_last_value(cc_times, [float(v) for v in cc_vals], start)))
                        if (include_cc and cc_times)
                        else None
                    )
                    cc64_ratio = (
                        _active_ratio(cc_times, cc_vals, start, end)
                        if include_cc
                        else None
                    )
                    cc11_on = (
                        int(round(_last_value(cc11_times, [float(v) for v in cc11_vals], start)))
                        if (include_cc11 and cc11_times)
                        else None
                    )
                    cc11_mean = (
                        _mean_value(cc11_times, cc11_vals, start, end)
                        if include_cc11
                        else None
                    )
                    if include_bend:
                        bend = (
                            int(round(_last_value(bend_times, [float(v) for v in bend_vals], start)))
                            if bend_times
                            else None
                        )
                        bend_range = _last_value(range_times, range_vals, start)
                        samples = _sample_bend(bend_times, bend_vals, start, end)
                        if samples:
                            semis = pb_math.pb_to_semi(samples, bend_range)
                            bend_max_semi = max(abs(s) for s in semis)
                            bend_rms_semi = math.sqrt(
                                sum(s * s for s in semis) / len(semis)
                            )
                            vib_rate_hz = _vibrato_rate(
                                samples, end - start, bend_range
                            )
                            if not math.isfinite(bend_max_semi):
                                bend_max_semi = None
                            if not math.isfinite(bend_rms_semi):
                                bend_rms_semi = None
                            if vib_rate_hz is not None and not math.isfinite(
                                vib_rate_hz
                            ):
                                vib_rate_hz = None
                        else:
                            bend_max_semi = bend_rms_semi = vib_rate_hz = None
                    else:
                        bend = bend_range = bend_max_semi = bend_rms_semi = (
                            vib_rate_hz
                        ) = None
                    rows.append(
                        NoteRow(
                            pitch=note.pitch,
                            duration=end - start,
                            bar=bar,
                            position=position,
                            velocity=note.velocity,
                            program=prog_val,
                            chord_symbol="",
                            articulation="",
                            q_onset=q_onset,
                            q_duration=q_duration,
                            cc64=cc64,
                            cc64_ratio=cc64_ratio,
                            cc11_at_onset=cc11_on,
                            cc11_mean=cc11_mean,
                            bend=bend,
                            bend_range=bend_range,
                            bend_max_semi=bend_max_semi,
                            bend_rms_semi=bend_rms_semi,
                            vib_rate_hz=vib_rate_hz,
                        )
                    )

        if include_bend and bend_events is not None:
            bend_events.mkdir(parents=True, exist_ok=True)
            ev_path = bend_events / f"{path.stem}_bend_events.csv"
            with ev_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["time", "value_14bit", "semitone", "track"])
                process(writer)
        else:
            process(None)
        if include_cc11 and cc_events is not None:
            cc_events.mkdir(parents=True, exist_ok=True)
            ev_path = cc_events / f"{path.stem}_cc11_events.csv"
            with ev_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["time", "value", "track"])
                for track_idx, inst in enumerate(pm.instruments):
                    for cc in inst.control_changes:
                        if cc.number == 11:
                            writer.writerow([float(cc.time), cc.value, track_idx])
    return rows


def build_note_csv(
    src: Path,
    out: Path,
    include_cc: bool = True,
    include_bend: bool = True,
    bend_events: Optional[Path] = None,
    *,
    include_cc11: bool = False,
    cc_events: Optional[Path] = None,
) -> None:
    """Extract rich note data from ``src`` MIDI folder into ``out`` CSV."""
    midi_paths = sorted(src.rglob("*.mid"))
    rows = scan_midi_files(
        midi_paths,
        include_cc,
        include_bend,
        bend_events,
        include_cc11=include_cc11,
        cc_events=cc_events,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "pitch",
        "duration",
        "bar",
        "position",
        "velocity",
        "program",
        "chord_symbol",
        "articulation",
        "q_onset",
        "q_duration",
    ]
    if include_cc:
        headers.extend(["CC64", "cc64_ratio"])
    if include_cc11:
        # Ensure CC11 columns are always emitted even when no CC events exist
        headers.extend(["cc11_at_onset", "cc11_mean"])
    if include_bend:
        headers.extend(
            ["bend", "bend_range", "bend_max_semi", "bend_rms_semi", "vib_rate_hz"]
        )

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            if include_cc:
                data["CC64"] = data.pop("cc64")
                data["cc64_ratio"] = data.pop("cc64_ratio")
            else:
                data.pop("cc64")
                data.pop("cc64_ratio")
            if not include_cc11:
                data.pop("cc11_at_onset")
                data.pop("cc11_mean")
            else:
                pass
            if not include_bend:
                data.pop("bend")
                data.pop("bend_range")
                data.pop("bend_max_semi")
                data.pop("bend_rms_semi")
                data.pop("vib_rate_hz")
            else:
                data["bend_range"] = _format_bend_range(data["bend_range"])
            writer.writerow(data)


def coverage_stats(csv_path: Path) -> None:
    """Print percentage of non-null values for each column using pandas."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    total = len(df)
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / total * 100) if total else 0
        print(f"{col}: {pct:.1f}% coverage ({non_null}/{total})")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build rich note CSV from MIDI files")
    p.add_argument("src", type=Path, nargs="?", help="Directory containing MIDI files")
    p.add_argument("--out", type=Path, help="Output CSV path")
    p.add_argument("--no-cc", action="store_true", help="Exclude sustain pedal column")
    p.add_argument("--no-bend", action="store_true", help="Exclude pitch bend column")
    p.add_argument(
        "--include-cc11",
        action="store_true",
        help="Include CC11 onset/mean columns",
    )
    p.add_argument(
        "--coverage", type=Path, help="Compute coverage stats for an existing CSV"
    )
    p.add_argument(
        "--emit-bend-events",
        type=Path,
        help="Directory to write raw pitch-bend events as CSV",
    )
    p.add_argument(
        "--emit-cc-events",
        type=Path,
        help="Directory to write raw CC11 events as CSV",
    )
    args = p.parse_args(argv)

    if args.coverage:
        coverage_stats(args.coverage)
        return 0

    if args.src is None or args.out is None:
        p.error("src and --out required unless --coverage is specified")

    build_note_csv(
        args.src,
        args.out,
        include_cc=not args.no_cc,
        include_bend=not args.no_bend,
        bend_events=args.emit_bend_events,
        include_cc11=args.include_cc11,
        cc_events=args.emit_cc_events,
    )
    print(f"✅ Rich note CSV generated: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
