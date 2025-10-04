#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import mido  # type: ignore[import]
import pretty_midi  # type: ignore[import]
from mido.midifiles import meta as mido_meta  # type: ignore[import]

# RuntimeWarning を抑制（tempo on non-zero tracks など）
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="pretty_midi.pretty_midi",
)

PrettyMIDIExceptionT = cast(
    type[Exception],
    getattr(pretty_midi, "PrettyMIDIException", RuntimeError),
)

LOGGER = logging.getLogger(__name__)


def _load_pretty_midi_safe(path: Path) -> pretty_midi.PrettyMIDI | None:
    """Load MIDI robustly:
    1) try PrettyMIDI directly
    2) if it fails, reparse via mido with clip=True and reserialize
    3) if KeySignatureError persists, monkey-patch mido's meta builder
    4) on failure, return None
    """
    KeySigError = getattr(mido_meta, "KeySignatureError", Exception)

    try:
        # 1) まず素直に読み込む（速い）
        return pretty_midi.PrettyMIDI(str(path))
    except (OSError, ValueError, RuntimeError, EOFError, KeySigError):
        pass
    try:
        # 2) 壊れを mido(clip=True) で矯正 → バッファへ保存
        with open(path, "rb") as f:
            raw = f.read()
        mid = mido.MidiFile(file=io.BytesIO(raw), clip=True)
        buf = io.BytesIO()
        mid.save(file=buf)  # type: ignore[arg-type]
        buf.seek(0)
        return pretty_midi.PrettyMIDI(buf)
    except EOFError:
        LOGGER.warning("Truncated MIDI (EOFError) skipped: %s", path)
    except KeySigError:
        # 3) KeySignatureError: patch builder for key_signature and retry
        orig_build = mido_meta.build_meta_message  # type: ignore[misc]

        def safe_build_meta_message(meta_type: int, data: bytes, delta: int) -> Any:
            try:
                return orig_build(meta_type, data, delta)  # type: ignore[misc]
            except KeySigError:
                if meta_type == 0x59:  # key_signature
                    # 代替として C major を挿入（time は保持）
                    return mido.MetaMessage(  # type: ignore[call-arg]
                        "key_signature",
                        key="C",
                        time=delta,
                    )
                raise

        mido_meta.build_meta_message = safe_build_meta_message  # type: ignore[assignment]
        try:
            mid = mido.MidiFile(filename=str(path), clip=True)
            buf = io.BytesIO()
            mid.save(file=buf)  # type: ignore[arg-type]
            buf.seek(0)
            return pretty_midi.PrettyMIDI(buf)
        except (OSError, ValueError) as e:
            LOGGER.warning(
                "Bad MIDI skipped after key_signature patch (%s): %s",
                type(e).__name__,
                path,
            )
        finally:
            # 必ず元に戻す
            mido_meta.build_meta_message = orig_build  # type: ignore[assignment]
    except (OSError, ValueError) as e:
        LOGGER.warning(
            "Bad MIDI skipped (%s): %s",
            type(e).__name__,
            path,
        )
    return None


def midi_stats(path: Path) -> Dict[str, Any] | None:
    """Compute lightweight MIDI statistics for a PrettyMIDI file."""
    midi = _load_pretty_midi_safe(path)
    if midi is None:
        return None

    beats = midi.get_beats()
    _, tempi = midi.get_tempo_changes()

    midi_any: Any = midi
    time_signatures = cast(
        Sequence[Any],
        getattr(midi_any, "time_signature_changes", []),
    )
    instruments = cast(
        Sequence[Any],
        getattr(midi_any, "instruments", []),
    )

    note_count = 0
    note_durations: list[float] = []
    duration = 0.0
    for instrument in instruments:
        notes = cast(Sequence[Any], getattr(instrument, "notes", []))
        note_count += len(notes)
        for note in notes:
            end = float(getattr(note, "end", 0.0))
            start = float(getattr(note, "start", 0.0))
            note_durations.append(end - start)
            if end > duration:
                duration = end

    meta: Dict[str, Any] = {
        "tempo": float(tempi[0]) if tempi.size else None,
        "tempo_count": int(tempi.size),
        "beats": int(beats.size),
        "notes": int(note_count),
        "duration_sec": float(duration),
    }

    if time_signatures:
        ts = time_signatures[0]
        numerator = getattr(ts, "numerator", None)
        denominator = getattr(ts, "denominator", None)
        if numerator is not None and denominator is not None:
            meta["time_signature"] = f"{numerator}/{denominator}"
        else:
            meta["time_signature"] = None
        meta["time_signature_count"] = len(time_signatures)
    else:
        meta["time_signature"] = None
        meta["time_signature_count"] = 0

    if note_durations:
        avg_dur = sum(note_durations) / len(note_durations)
        meta["avg_note_duration"] = float(avg_dur)
    else:
        meta["avg_note_duration"] = None

    return meta


def enrich_manifest(input_path: Path, output_path: Path) -> None:
    """Read a JSONL manifest, append MIDI statistics, and write the result."""
    processed = 0
    failures = 0
    skipped = 0

    with (
        input_path.open("r", encoding="utf-8") as reader,
        output_path.open("w", encoding="utf-8") as writer,
    ):
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            path = record.get("path")
            try:
                if not path:
                    raise ValueError("record does not contain 'path' field")
                stats = midi_stats(Path(path))
                if stats is None:
                    # 壊れMIDIは飛ばして継続
                    skipped += 1
                    error_msg = "corrupted or truncated MIDI"
                    record.setdefault("meta", {}).update({"error": error_msg})
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
            except PrettyMIDIExceptionT as exc:
                failures += 1
                record.setdefault("meta", {}).update({"error": str(exc)})
                LOGGER.warning("Failed to compute stats for %s: %s", path, exc)
            except (OSError, ValueError) as exc:
                failures += 1
                record.setdefault("meta", {}).update({"error": str(exc)})
                LOGGER.warning("Failed to compute stats for %s: %s", path, exc)
            else:
                record.setdefault("meta", {}).update(stats)
                processed += 1
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Processed %d records (failures=%d, skipped=%d)",
        processed,
        failures,
        skipped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich a manifest file with MIDI stats")
    parser.add_argument("in_jsonl", type=Path, help="Input JSONL manifest")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.in_jsonl.exists():
        raise SystemExit(f"Input manifest not found: {args.in_jsonl}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enrich_manifest(args.in_jsonl, args.out)


if __name__ == "__main__":
    main()
