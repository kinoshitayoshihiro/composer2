from __future__ import annotations

"""Simple configuration validator for UJAM bridge mappings."""

import argparse
import json
import pathlib
from typing import Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .ujam_map import MAP_DIR, _parse_simple, _validate_map


def _load(path: pathlib.Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_simple(text)


def validate(path: pathlib.Path) -> List[str]:
    """Validate mapping *path* and return a list of problems."""
    data = _load(path)
    return _validate_map(data)


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
