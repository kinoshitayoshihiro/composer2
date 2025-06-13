#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare section labels between chordmap and config.

This script loads two YAML files:
1. `data/processed_chordmap_with_emotion.yaml`
   (or a custom path given as the first CLI argument)
2. `config/main_cfg.yml`
   (or a custom path given as the second CLI argument)

It lists section names that exist only in one of the files.

Example:
    $ python tools/compare_sections.py
    --- Sections in chordmap but not in config ---
    Verse 1
    ...

    $ python tools/compare_sections.py my_chordmap.yaml my_cfg.yml
"""
from __future__ import annotations

from pathlib import Path
import sys
import yaml


def load_chordmap_sections(path: Path) -> set[str]:
    """Return the set of section names from the chordmap."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sections = data.get("sections", {})
    return set(sections.keys()) if isinstance(sections, dict) else set()


def load_cfg_sections(path: Path) -> set[str]:
    """Return the set of sections listed in the config."""
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sections = cfg.get("sections_to_generate", [])
    return set(sections)


def main() -> None:
    chordmap_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed_chordmap_with_emotion.yaml")
    cfg_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/main_cfg.yml")

    chordmap_sections = load_chordmap_sections(chordmap_path)
    cfg_sections = load_cfg_sections(cfg_path)

    only_chordmap = sorted(chordmap_sections - cfg_sections)
    only_cfg = sorted(cfg_sections - chordmap_sections)

    print("--- Sections in chordmap but not in config ---")
    for name in only_chordmap:
        print(name)

    print("--- Sections in config but not in chordmap ---")
    for name in only_cfg:
        print(name)


if __name__ == "__main__":
    main()
