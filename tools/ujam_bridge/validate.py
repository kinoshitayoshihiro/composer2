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
    loader = yaml
    if loader is None:
        try:
            import yaml as _yaml  # type: ignore
        except Exception:
            _yaml = None  # type: ignore
        loader = _yaml
        if loader is not None:
            globals()["yaml"] = loader
    if loader is not None:
        data = loader.safe_load(text)
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), {})
        if not isinstance(data, dict):
            data = _parse_simple(text)
    else:
        data = _parse_simple(text)
    if not isinstance(data, dict):
        data = {}
    return data


def _validate_keyswitch_range(data: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(data, dict):
        return issues

    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    play_raw = data.get("play_range")
    play_range = play_raw if isinstance(play_raw, dict) else {}
    play_low_raw = _coerce_int(play_range.get("low"), KS_MIN)
    play_high_raw = _coerce_int(play_range.get("high"), KS_MAX)
    if play_low_raw > play_high_raw:
        play_low_raw, play_high_raw = play_high_raw, play_low_raw
    play_low = max(0, min(play_low_raw, 127))
    play_high = max(play_low, min(play_high_raw, 127))
    ks_low = play_low if play_range else KS_MIN
    ks_high = play_high if play_range else KS_MAX

    ks_dict = data.get("keyswitch")
    if isinstance(ks_dict, dict):
        for name, value in ks_dict.items():
            if not isinstance(name, str):
                continue
            try:
                pitch = int(value)
            except Exception:
                issues.append(f"keyswitch '{name}' note {value!r} is not an integer")
                continue
            if pitch < 0 or pitch > 127:
                issues.append(
                    f"keyswitch '{name}' out of MIDI range 0..127 (got {pitch})"
                )
            elif pitch < ks_low or pitch > ks_high:
                issues.append(
                    f"keyswitch '{name}' out of range {ks_low}..{ks_high} (got {pitch})"
                )

    ks_list = data.get("keyswitches")
    if isinstance(ks_list, list):
        for idx, item in enumerate(ks_list):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            label = name if isinstance(name, str) and name else f"#{idx}"
            note = item.get("note")
            try:
                pitch = int(note)
            except Exception:
                issues.append(f"keyswitch '{label}' note {note!r} is not an integer")
                continue
            if pitch < 0 or pitch > 127:
                issues.append(
                    f"keyswitch '{label}' out of MIDI range 0..127 (got {pitch})"
                )
            elif pitch < ks_low or pitch > ks_high:
                issues.append(
                    f"keyswitch '{label}' out of range {ks_low}..{ks_high} (got {pitch})"
                )
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
        play_low = low
        play_high = high
        ks_low = min(play_low, KS_MIN)
        ks_high = max(play_high, KS_MAX)

        def _check_range(name: str, value: object) -> None:
            try:
                note = int(value)  # type: ignore[arg-type]
            except Exception:
                return
            if note < ks_low or note > ks_high:
                _emit(f"keyswitch '{name}' out of range {ks_low}..{ks_high} (got {note})")
            elif play_defined and play_low <= note <= play_high:
                _emit(
                    f"keyswitch '{name}' overlaps play range {play_low}..{play_high} (got {note})"
                )

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
