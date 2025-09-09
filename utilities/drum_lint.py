from __future__ import annotations

"""Lint drum pattern files for unknown instrument names.

This utility scans ``drum_patterns.yml`` and ``rhythm_library.yml`` and warns
if any event uses an instrument name not present in
:data:`generator.drum_generator.GM_DRUM_MAP`.
"""

from pathlib import Path
import logging
from typing import Iterable, Set

import yaml

from generator.drum_generator import GM_DRUM_MAP
from utilities.fill_dsl import parse_fill_dsl

logger = logging.getLogger(__name__)


def check_drum_patterns(path: Path = Path("data/drum_patterns.yml")) -> Set[str]:
    """Check drum pattern YAML for unknown instruments.

    Parameters
    ----------
    path:
        Path to ``drum_patterns.yml``.

    Returns
    -------
    Set[str]
        Set of unknown instrument names.
    """

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    patterns = data.get("drum_patterns", {})
    unknown: Set[str] = set()

    for pat_def in patterns.values():
        pat = pat_def.get("pattern", [])
        events: Iterable[dict]
        if isinstance(pat, str):
            try:
                events = parse_fill_dsl(pat)
            except Exception as exc:  # pragma: no cover - parsing errors reported
                logger.warning("Failed to parse DSL pattern: %s", exc)
                continue
        else:
            events = pat
        for event in events:
            if not isinstance(event, dict):
                continue
            inst = event.get("instrument")
            if inst and inst not in GM_DRUM_MAP:
                unknown.add(inst)

    for inst in sorted(unknown):
        logger.warning("Unknown drum instrument '%s'", inst)

    return unknown


def _check_events(events: Iterable[dict]) -> Set[str]:
    """Return unknown drum instruments from a sequence of events."""

    unknown: Set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        inst = ev.get("instrument")
        if inst and inst not in GM_DRUM_MAP:
            unknown.add(inst)
    return unknown


def check_rhythm_library(path: Path = Path("data/rhythm_library.yml")) -> Set[str]:
    """Check ``rhythm_library.yml`` for unknown drum instruments."""

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    patterns = data.get("drum_patterns", {})
    unknown: Set[str] = set()

    for pat_def in patterns.values():
        unknown.update(_check_events(pat_def.get("pattern", [])))
        for fill in pat_def.get("fill_ins", {}).values():
            unknown.update(_check_events(fill))

    for inst in sorted(unknown):
        logger.warning("Unknown drum instrument '%s'", inst)

    return unknown


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Lint drum pattern instruments in YAML files"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/drum_patterns.yml"),
        help="Path to drum_patterns.yml",
    )
    parser.add_argument(
        "--rhythm-file",
        type=Path,
        default=None,
        help="Path to rhythm_library.yml",
    )
    args = parser.parse_args()

    unknown = check_drum_patterns(args.file)
    if args.rhythm_file:
        unknown |= check_rhythm_library(args.rhythm_file)

    if unknown:
        logger.warning("Found %d unknown instruments", len(unknown))
    else:
        print("No issues found.")
