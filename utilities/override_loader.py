# -*- coding: utf-8 -*-
"""
override_loader.py – Refactored for Pydantic v2
================================================
Provides utilities to load, validate, and query per-section arrangement overrides.

Features:
- JSON / YAML / TOML support
- Caching to avoid repeated I/O
- Strict validation of section names and part overrides
- Simple API: load_overrides(), get_part_override()

Dependencies:
    pip install pyyaml tomli pydantic>=2.6

Usage:
    from override_loader import load_overrides, get_part_override
    overrides_model = load_overrides("data/arrangement_overrides.json")
    guitar_cfg_model = get_part_override(overrides_model, section="Chorus 1", part="guitar")
    guitar_params_dict = guitar_cfg_model.model_dump(exclude_unset=True) # To get a dict
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, List # List を追加
import tomli
import yaml
import io
import logging
from pydantic import BaseModel, Field, RootModel, ValidationError

logger = logging.getLogger(__name__)

class PartOverride(BaseModel):
    # Common overrides
    rhythm_key: Optional[str] = None
    rhythm_key_rh: Optional[str] = None
    rhythm_key_lh: Optional[str] = None
    velocity: Optional[int] = Field(None, ge=1, le=127)
    humanize_opt: Optional[bool] = None
    template_name: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None
    velocity_shift: Optional[int] = None # ★★★ BassGenerator用に追加 ★★★
    velocity_shift_on_kick: Optional[int] = None  # default 10
    swing_ratio: Optional[float] = None
    swing_ratio_rh: Optional[float] = None
    swing_ratio_lh: Optional[float] = None
    offset_profile: Optional[str] = None
    offset_profile_rh: Optional[str] = None
    offset_profile_lh: Optional[str] = None

    # Guitar specific
    palm_mute: Optional[bool] = None
    humanize_timing_sec: Optional[float] = None
    strum_direction_cycle: Optional[str] = None

    # Bass specific
    weak_beat_style: Optional[str] = None
    approach_on_4th_beat: Optional[bool] = None
    approach_style_on_4th: Optional[str] = None

    # Piano specific
    weak_beat_style_rh: Optional[str] = None
    weak_beat_style_lh: Optional[str] = None
    fill_on_4th: Optional[bool] = None
    fill_length_beats: Optional[float] = None
    velocity_shift_rh: Optional[int] = None
    velocity_shift_lh: Optional[int] = None

    # Drum specific
    ghost_hat_on_offbeat: Optional[bool] = None
    additional_kick_density: Optional[float] = None
    # drum_style_key: Optional[str] = None # Covered by rhythm_key
    # drum_base_velocity: Optional[int] = None # Covered by velocity
    # drum_fill_interval_bars: Optional[int] = None
    # drum_fill_keys: Optional[List[str]] = None

    options: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class SectionOverride(BaseModel):
    guitar: Optional[PartOverride] = None
    bass: Optional[PartOverride] = None
    drums: Optional[PartOverride] = None
    piano: Optional[PartOverride] = None
    model_config = {"extra": "allow"}


class Overrides(RootModel[Dict[str, SectionOverride]]):
    root: Dict[str, SectionOverride]

    def get_section(self, section_name: str) -> Optional[SectionOverride]:
        return self.root.get(section_name)


_OVERRIDES_CACHE: Dict[Path, Overrides] = {}

def load_overrides(path: str | Path, *, force_reload: bool = False) -> Overrides:
    p = Path(path)
    if not force_reload and p in _OVERRIDES_CACHE:
        logger.debug(f"Returning cached overrides for {p}")
        return _OVERRIDES_CACHE[p]

    if not p.exists():
        logger.warning(f"Override file not found: {p}. Returning empty Overrides model.")
        return Overrides(root={})

    text_content = p.read_text(encoding='utf-8')
    if not text_content.strip():
        logger.warning(f"Override file is empty: {p}. Returning empty Overrides model.")
        return Overrides(root={})

    data: Dict[str, Any]
    logger.info(f"Loading overrides from: {p}")
    if p.suffix == ".json":
        data = json.loads(text_content)
    elif p.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text_content)
    elif p.suffix == ".toml":
        data = tomli.loads(text_content)
    else:
        raise ValueError(f"Unsupported override file format: {p.suffix}")

    try:
        ov = Overrides.model_validate(data)
        logger.info(f"Successfully validated overrides from {p}")
    except ValidationError as e:
        logger.error(f"Override validation failed for {p}: {e}")
        raise ValueError(f"Override validation failed for {p}: {e}")

    _OVERRIDES_CACHE[p] = ov
    return ov

def get_part_override(
    overrides_model: Overrides,
    section: str,
    part: str,
) -> PartOverride:
    if not overrides_model or not overrides_model.root:
        return PartOverride()

    section_data_model = overrides_model.root.get(section)

    if section_data_model and isinstance(section_data_model, SectionOverride):
        part_data_model = getattr(section_data_model, part, None)
        if part_data_model and isinstance(part_data_model, PartOverride):
            return part_data_model
        else:
            logger.debug(f"No override found for part '{part}' in section '{section}'.")
    else:
        logger.debug(f"No override section found for '{section}'.")

    return PartOverride()

if __name__ == "__main__":
    import argparse
    import pprint
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="Validate and inspect arrangement overrides.")
    parser.add_argument("file", type=Path, help="Path to arrangement_overrides.json/yml/toml")
    parser.add_argument("--list-sections", action="store_true", help="List all section names found in the override file.")
    parser.add_argument("--section", type=str, help="Section name to inspect.")
    parser.add_argument("--part", type=str, help="Part name (e.g., guitar, bass, drums, piano) to inspect within the section.")
    args = parser.parse_args()
    try:
        ov_model = load_overrides(args.file)
        if args.list_sections:
            if ov_model and ov_model.root:
                print("Sections found:", list(ov_model.root.keys()))
            else:
                print("No sections found in the override model.")
        elif args.section and args.part:
            cfg_part_model = get_part_override(ov_model, args.section, args.part)
            print(f"\nOverrides for Section: '{args.section}', Part: '{args.part}':")
            pprint.pprint(cfg_part_model.model_dump(exclude_unset=True))
        elif args.section and not args.part:
            section_data = ov_model.get_section(args.section)
            if section_data:
                print(f"\nOverrides for Section: '{args.section}':")
                pprint.pprint(section_data.model_dump(exclude_unset=True))
            else:
                print(f"Section '{args.section}' not found in overrides.")
        else:
            print("Please specify an action: --list-sections, or --section <name> --part <name>, or --section <name>")
            print("\nFull override model structure:")
            pprint.pprint(ov_model.model_dump(exclude_unset=True) if ov_model else {})
    except FileNotFoundError as e_fnf: print(f"Error: {e_fnf}")
    except ValueError as e_val: print(f"Error: {e_val}")
    except Exception as e_generic: print(f"An unexpected error occurred: {e_generic}")