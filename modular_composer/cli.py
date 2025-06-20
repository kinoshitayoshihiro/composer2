from __future__ import annotations

import argparse
import importlib.metadata as _md
from pathlib import Path

from utilities.groove_sampler_v2 import generate_events, save, train  # noqa: F401

__version__ = _md.version("modular_composer")


def _cmd_demo(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose demo")
    ap.add_argument("-o", "--out", type=Path, default=Path("demo.mid"))
    ns = ap.parse_args(args)
    model = train(Path("data/loops"), n=2, auto_res=True)
    ev = generate_events(model, bars=4)
    # (PrettyMIDI export placeholder)
    print(f"[demo] generated {len(ev)} events -> {ns.out}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: modcompose <command> [<args>]\n\ncommands: demo")
        sys.exit(0)
    cmd, *rest = argv
    if cmd == "demo":
        _cmd_demo(rest)
    else:
        sys.exit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
