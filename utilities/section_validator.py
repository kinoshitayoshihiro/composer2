"""Utilities to validate structural sections against MIDI files.

This module reads a YAML file describing song sections and validates that the
exported MIDI files for each section actually span the declared bar range.  It
is useful to ensure that an arrangement's structural metadata matches the MIDI
exports used downstream in the pipeline (e.g. for VOCALOID / Synthesizer V
import).

The expected YAML format is::

    sections:
      - label: Verse
        midi: verse.mid
        start_bar: 1
        end_bar: 8
      - label: Chorus
        midi: chorus.mid
        start_bar: 9
        end_bar: 16

``midi`` paths are resolved relative to the YAML file's directory.  Bar numbers
are 1-based and inclusive.

Additional section labels can be allowed via either the environment variable
``SECTION_VALIDATOR_LABELS`` (comma-separated) or an ``--extra-labels`` command
line argument when running this module as a script.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pretty_midi
import yaml

__all__ = ["validate_sections", "VALID_LABELS"]

logger = logging.getLogger(__name__)

# Allowed structural labels.  Additional labels can be added via environment
# variable ``SECTION_VALIDATOR_LABELS`` or ``--extra-labels`` CLI option.
VALID_LABELS = {"Verse", "Pre-Chorus", "Chorus", "Bridge"}


class SectionValidationError(ValueError):
    """Raised when a section fails validation."""


def _midi_bar_range(path: Path) -> tuple[int, int]:
    """Return the (start_bar, end_bar) for the notes in ``path``.

    Supports multiple tempo and time-signature changes by integrating the bar
    length piecewise across the MIDI timeline.  Raises
    :class:`SectionValidationError` if the MIDI contains no notes.
    """

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        raise SectionValidationError(f"{path.name} contains no notes")

    start_time = min(n.start for n in notes)
    end_time = max(n.end for n in notes)

    # ``pretty_midi`` in this repository has a patched ``get_tempo_changes`` that
    # returns ``(tempi, times)`` instead of ``(times, tempi)``.  Detect both
    # orderings to remain compatible if upstream behaviour changes.
    a, b = pm.get_tempo_changes()
    a = list(a)
    b = list(b)
    if a and a[0] <= 1e-6 and all(x <= y for x, y in zip(a, a[1:])):
        tempo_times, tempi = a, b
    else:  # patched ordering
        tempi, tempo_times = a, b
    if not tempo_times:
        tempo_times = [0.0]
        tempi = [120.0]
    elif tempo_times[0] != 0.0:
        tempo_times.insert(0, 0.0)
        tempi.insert(0, tempi[0])

    ts_changes = [ts for ts in pm.time_signature_changes]
    if not ts_changes or ts_changes[0].time != 0.0:
        ts_changes.insert(0, pretty_midi.TimeSignature(4, 4, 0.0))

    def seconds_per_bar(bpm: float, ts: pretty_midi.TimeSignature) -> float:
        return ts.numerator * (60.0 / bpm) * (4.0 / ts.denominator)

    def time_to_bar(time: float) -> int:
        bar = 0.0
        cur_t = 0.0
        tempo_idx = 0
        ts_idx = 0
        while cur_t < time:
            next_tempo = (
                tempo_times[tempo_idx + 1] if tempo_idx + 1 < len(tempo_times) else time
            )
            next_ts = (
                ts_changes[ts_idx + 1].time if ts_idx + 1 < len(ts_changes) else time
            )
            boundary = min(next_tempo, next_ts, time)
            spb = seconds_per_bar(float(tempi[tempo_idx]), ts_changes[ts_idx])
            bar += (boundary - cur_t) / spb
            cur_t = boundary
            if cur_t == next_tempo and tempo_idx + 1 < len(tempo_times):
                tempo_idx += 1
            if cur_t == next_ts and ts_idx + 1 < len(ts_changes):
                ts_idx += 1
        return int(bar) + 1

    start_bar = time_to_bar(start_time)
    end_bar = time_to_bar(end_time - 1e-9)
    return start_bar, end_bar


def _valid_labels(extra_labels: list[str] | None = None) -> set[str]:
    labels = set(VALID_LABELS)
    env = os.environ.get("SECTION_VALIDATOR_LABELS")
    if env:
        labels.update(l.strip() for l in env.split(",") if l.strip())
    if extra_labels:
        labels.update(extra_labels)
    return labels


def validate_sections(
    structure_path: Path | str, extra_labels: list[str] | None = None
) -> bool:
    """Validate that section metadata matches the exported MIDI files.

    Parameters
    ----------
    structure_path:
        Path to a YAML file describing sections.  See module level documentation
        for the expected format.
    extra_labels:
        Optional list of additional section labels to allow.

    Returns
    -------
    bool
        ``True`` if all sections are valid.  ``SectionValidationError`` is raised
        on the first failure encountered.
    """

    structure_path = Path(structure_path)
    with structure_path.open("r", encoding="utf8") as fh:
        data = yaml.safe_load(fh) or {}

    allowed_labels = _valid_labels(extra_labels)
    sections = data.get("sections", [])
    if not isinstance(sections, list):  # pragma: no cover - defensive
        raise SectionValidationError("'sections' must be a list")

    for entry in sections:
        label = entry.get("label")
        midi_rel = entry.get("midi")
        start_bar = entry.get("start_bar")
        end_bar = entry.get("end_bar")

        if label not in allowed_labels:
            raise SectionValidationError(f"unknown label '{label}'")

        if midi_rel is None or start_bar is None or end_bar is None:
            raise SectionValidationError(
                f"section '{label}' is missing required fields"
            )

        midi_path = (structure_path.parent / midi_rel).resolve()
        logger.debug(
            "Validating section '%s' using MIDI '%s' (bars %s-%s)",
            label,
            midi_path,
            start_bar,
            end_bar,
        )
        actual_start, actual_end = _midi_bar_range(midi_path)
        if (actual_start, actual_end) != (start_bar, end_bar):
            raise SectionValidationError(
                f"section '{label}' expected bars {start_bar}-{end_bar} but MIDI "
                f"spans {actual_start}-{actual_end}"
            )

    return True


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path, help="Path to structure YAML")
    parser.add_argument(
        "--extra-labels",
        help="Comma-separated additional labels to allow",
        default="",
    )
    args = parser.parse_args(argv)
    extra = [l.strip() for l in args.extra_labels.split(",") if l.strip()]
    validate_sections(args.structure, extra_labels=extra)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
