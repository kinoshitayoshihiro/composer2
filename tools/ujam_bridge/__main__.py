from __future__ import annotations

"""Unified CLI entry point for UJAM bridge utilities."""

import argparse
import pathlib
from typing import List

from . import gen_staircase, validate


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.ujam_bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate product maps")
    v.add_argument("--all", action="store_true")
    v.add_argument("--product")
    v.add_argument("--strict", action="store_true")
    v.add_argument("--report", type=pathlib.Path)
    v.set_defaults(func=lambda args: validate._cmd_validate(args))

    g = sub.add_parser("gen-staircase", help="generate keyswitch staircase")
    g.add_argument("--product", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--note-len", type=float, default=1.0)
    g.add_argument("--gap", type=float, default=0.1)
    g.add_argument("--tempo", type=float, default=120.0)
    g.add_argument("--ppq", type=int, default=480)
    g.add_argument("--channel", type=int, default=0)
    g.add_argument("--velocity", type=int, default=100)

    def _run_gen(args: argparse.Namespace) -> int:
        gen_staircase.generate(
            args.product,
            pathlib.Path(args.out),
            note_len=args.note_len,
            gap=args.gap,
            tempo=args.tempo,
            ppq=args.ppq,
            channel=args.channel,
            velocity=args.velocity,
        )
        return 0

    g.set_defaults(func=_run_gen)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
