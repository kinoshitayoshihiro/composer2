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


def validate(path: pathlib.Path) -> List[str]:
    """Validate mapping *path* and return a list of problems."""
    data = _load(path)
    issues = _validate_map(data)

    pr = data["play_range"] if isinstance(data, dict) and isinstance(data.get("play_range"), dict) else {}
    ks_dict = data["keyswitch"] if isinstance(data, dict) and isinstance(data.get("keyswitch"), dict) else {}
    ks_raw = data.get("keyswitch") if isinstance(data, dict) else None

    try:
        low = int(pr.get("low", KS_MIN))
        high = int(pr.get("high", KS_MAX))
        if low < KS_MIN or high > KS_MAX:
            issues.append(f"play_range out of range {KS_MIN}..{KS_MAX} (got {low}..{high})")
    except Exception:
        pass

    def _check(name: str, value: object) -> None:
        try:
            note = int(value)  # type: ignore[arg-type]
        except Exception:
            return
        if not (KS_MIN <= note <= KS_MAX):
            issues.append(f"keyswitch '{name}' out of range {KS_MIN}..{KS_MAX} (got {note})")

    for name, value in ks_dict.items():
        if isinstance(name, str):
            _check(name, value)

    if isinstance(ks_raw, list):
        for item in ks_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str):
                _check(name, item.get("note"))

    ks_list = data.get("keyswitches")
    if isinstance(ks_list, list):
        for item in ks_list:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str):
                _check(name, item.get("note"))

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
