"""Mappings between General MIDI percussion numbers and names."""

import re


# Canonical General MIDI percussion names keyed by their numeric code.
NUM_TO_NAME: dict[int, str] = {
    35: "kick",  # Acoustic Bass Drum
    36: "kick",  # Bass Drum 1
    37: "sidestick",
    38: "snare",
    39: "clap",
    40: "snare_elec",
    41: "tom_low",
    42: "hh_closed",
    43: "tom_floor_h",
    44: "hh_pedal",
    45: "tom_mid_l",
    46: "hh_open",
    47: "tom_mid_h",
    48: "tom_high_l",
    49: "crash",
    50: "tom_high_h",
    51: "ride",
    52: "china",
    53: "ride_bell",
    54: "tambourine",
    55: "splash",
    56: "cowbell",
    57: "crash2",
    58: "vibraslap",
    59: "ride2",
    60: "bongo_h",
    61: "bongo_l",
    62: "conga_h_mute",
    63: "conga_h_open",
    64: "conga_l",
    65: "timbale_h",
    66: "timbale_l",
    67: "agogo_h",
    68: "agogo_l",
    69: "cabasa",
    70: "maracas",
    71: "whistle_h",
    72: "whistle_l",
    73: "guiro_s",
    74: "guiro_l",
    75: "claves",
    76: "woodblock_h",
    77: "woodblock_l",
    78: "cuica_mute",
    79: "cuica_open",
    80: "triangle_mute",
    81: "triangle_open",
}

# Alternative spellings mapped to canonical names.
ALIASES: dict[str, str] = {
    "chh": "hh_closed",
    "closed_hat": "hh_closed",
    "ohh": "hh_open",
    "open_hat": "hh_open",
    "phh": "hh_pedal",
    "pedal_hat": "hh_pedal",
}

NAME_TO_NUM: dict[str, int] = {v: k for k, v in NUM_TO_NAME.items()}


def number_to_name(num: int) -> str:
    """Return normalized name for a GM percussion number."""
    return NUM_TO_NAME.get(num, f"unk_{num:02d}")


def normalize_label(lbl: str | int) -> str:
    """Normalize an arbitrary label to a canonical GM name or ``unk_XX``."""
    if isinstance(lbl, int) or (isinstance(lbl, str) and lbl.isdigit()):
        name = number_to_name(int(lbl))
    else:
        name = str(lbl)
    name = ALIASES.get(name, name)
    if name in NAME_TO_NUM:
        return name
    m = re.match(r"unk_(\d{2})$", name)
    if m:
        return f"unk_{int(m.group(1)):02d}"
    return name


def label_to_number(lbl: str | int) -> int | None:
    """Resolve *lbl* to a GM percussion number if possible.

    ``lbl`` may be a numeric string, an alias, a canonical GM name, or an
    ``unk_XX`` placeholder.  Returns ``None`` when the label cannot be
    interpreted.
    """
    if isinstance(lbl, int):
        return lbl
    lbl = ALIASES.get(str(lbl), str(lbl))
    if lbl in NAME_TO_NUM:
        return NAME_TO_NUM[lbl]
    m = re.match(r"unk_(\d{2})$", str(lbl))
    return int(m.group(1)) if m else None

