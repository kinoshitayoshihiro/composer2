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

    # --- play_range の型を厳密に ---
    pr = data.get("play_range")
    if not isinstance(pr, dict):
        pr = {}

    def _to_int(value: object, default: int | None) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return default

    low = _to_int(pr.get("low"), KS_MIN)
    high = _to_int(pr.get("high"), KS_MAX)
    if low is None:
        low = KS_MIN
    if high is None:
        high = KS_MAX
    if low > high:
        low, high = high, low  # 入れ替えで自衛

    def _add_range_issue(name: str, pitch: object) -> None:
        note = _to_int(pitch, None)
        if note is None:
            issues.append(f"keyswitch '{name}' has non-integer pitch: {pitch}")
            return
        if not (low <= note <= high):
            msg = f"keyswitch '{name}' out of range {low}..{high} (got {note})"
            if msg not in issues:
                issues.append(msg)

    # --- keyswitch の検証 ---
    ks = data.get("keyswitch")
    if isinstance(ks, dict):
        for name, pitch in ks.items():
            if isinstance(name, str):
                _add_range_issue(name, pitch)
    elif isinstance(ks, list):
        for item in ks:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str):
                _add_range_issue(name, item.get("note"))

    ks_list = data.get("keyswitches")
    if isinstance(ks_list, list):
        for item in ks_list:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str):
                _add_range_issue(name, item.get("note"))

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
