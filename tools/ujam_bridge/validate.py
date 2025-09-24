from __future__ import annotations

"""Simple configuration validator for UJAM bridge mappings."""

import argparse
import json
import pathlib
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .ujam_map import KS_MIN, KS_MAX, MAP_DIR, _parse_simple, _validate_map


def _load(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _parse_simple(text)
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})
    if not isinstance(data, dict):
        data = {}
    return data


def _validate_keyswitch_range(data: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(data, dict):
        return issues

    def _check_note(label: str, value: object) -> int | None:
        if not isinstance(value, int):
            issues.append(f"keyswitch '{label}' note {value!r} is not an integer")
            return None
        note = int(value)
        if note < 0 or note > 127:
            issues.append(f"keyswitch '{label}' note {note} out of range 0..127")
        return note

    play_range = data.get("play_range")
    if isinstance(play_range, dict):
        try:
            low = int(play_range.get("low", KS_MIN))
        except Exception:
            low = KS_MIN
        try:
            high = int(play_range.get("high", KS_MAX))
        except Exception:
            high = KS_MAX
        if low > high:
            low, high = high, low
    else:
        low, high = KS_MIN, KS_MAX
    ks = data.get("keyswitch")
    if isinstance(ks, dict):
        for name, value in ks.items():
            if not isinstance(name, str):
                continue
            pitch = _check_note(name, value)
            if pitch is None:
                continue
            if pitch < low or pitch > high:
                issues.append(f"keyswitch '{name}' out of range {low}..{high} (got {pitch})")
    ks_list = data.get("keyswitches")
    if isinstance(ks_list, list):
        for idx, item in enumerate(ks_list):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            label = name if isinstance(name, str) and name else f"#{idx}"
            _check_note(label, item.get("note"))
    return issues


def validate(path: pathlib.Path) -> List[str]:
    """Validate mapping *path* and return a list of problems."""
    data = _load(path)
    issues = _validate_map(data)
    for msg in _validate_keyswitch_range(data):
        if msg not in issues:
            issues.append(msg)

    if isinstance(data, dict):
        play_range_raw = data.get("play_range")
        play_range = play_range_raw if isinstance(play_range_raw, dict) else {}
        play_defined = bool(play_range)

        def _emit(msg: str) -> None:
            if msg not in issues:
                issues.append(msg)

        try:
            low = int(play_range.get("low", KS_MIN))
        except Exception:
            low = KS_MIN
        try:
            high = int(play_range.get("high", KS_MAX))
        except Exception:
            high = KS_MAX
        raw_low, raw_high = low, high
        if raw_low < KS_MIN or raw_high > KS_MAX:
            _emit(f"play_range out of range {KS_MIN}..{KS_MAX} (got {raw_low}..{raw_high})")
        if low > high:
            low, high = high, low
        low = max(0, min(low, 127))
        high = max(low, min(high, 127))

        def _check_range(name: str, value: object) -> None:
            try:
                note = int(value)  # type: ignore[arg-type]
            except Exception:
                return
            if note < low or note > high:
                _emit(f"keyswitch '{name}' out of range {low}..{high} (got {note})")
            elif play_defined:
                _emit(f"keyswitch '{name}' overlaps play range {low}..{high} (got {note})")

        ks_dict = data.get("keyswitch")
        if isinstance(ks_dict, dict):
            for name, value in ks_dict.items():
                if isinstance(name, str):
                    _check_range(name, value)

        ks_raw = data.get("keyswitch")
        if isinstance(ks_raw, list):
            for item in ks_raw:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str):
                    _check_range(name, item.get("note"))

        ks_list = data.get("keyswitches")
        if isinstance(ks_list, list):
            for item in ks_list:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str):
                    _check_range(name, item.get("note"))

        rng_low, rng_high = KS_MIN, KS_MAX
        if isinstance(play_range_raw, dict):
            try:
                rng_low = int(play_range_raw.get("low", KS_MIN))
            except Exception:
                rng_low = KS_MIN
            try:
                rng_high = int(play_range_raw.get("high", KS_MAX))
            except Exception:
                rng_high = KS_MAX
        if rng_low > rng_high:
            rng_low, rng_high = rng_high, rng_low
        ks_map = data.get("keyswitch")
        if isinstance(ks_map, dict):
            for name, value in ks_map.items():
                if not isinstance(name, str):
                    continue
                try:
                    pitch = int(value)
                except Exception:
                    continue
                if pitch < rng_low or pitch > rng_high:
                    msg = f"keyswitch '{name}' pitch {pitch} out of range {rng_low}..{rng_high}"
                    if msg not in issues:
                        issues.append(msg)

    return issues


def _cmd_validate(args: argparse.Namespace) -> int:
    products: List[str]
    if args.all:
        products = [p.stem for p in MAP_DIR.glob("*.yaml")]
    elif args.product:
        products = [args.product]
    else:
        raise SystemExit("--all or --product required")
    report: Dict[str, List[str]] = {}
    for prod in products:
        path = MAP_DIR / f"{prod}.yaml"
        issues = validate(path)
        report[prod] = issues
        if issues:
            print(f"{prod}:")
            for msg in issues:
                print(f"  {msg}")
        else:
            print(f"{prod}: OK")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    if args.strict and any(report.values()):
        return 1
    return 0


def main(argv: List[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--product")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args(argv)
    return _cmd_validate(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
