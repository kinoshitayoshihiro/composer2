from __future__ import annotations

import argparse
import importlib.metadata as _md
from pathlib import Path

from utilities.groove_sampler_v2 import generate_events, load, save, train  # noqa: F401

__version__ = _md.version("modular_composer")


def _cmd_demo(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose demo")
    ap.add_argument("-o", "--out", type=Path, default=Path("demo.mid"))
    ns = ap.parse_args(args)
    model = train(Path("data/loops"), n=2, auto_res=True)
    ev = generate_events(model, bars=4)
    # (PrettyMIDI export placeholder)
    print(f"[demo] generated {len(ev)} events -> {ns.out}")


def _cmd_sample(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose sample")
    ap.add_argument("model", type=Path)
    ap.add_argument("-l", "--length", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    ap.add_argument(
        "--cond-kick", choices=["four_on_floor", "sparse"], default=None
    )
    ap.add_argument("-o", "--out", type=Path)
    ns = ap.parse_args(args)
    model = load(ns.model)
    ev = generate_events(
        model,
        bars=ns.length,
        temperature=ns.temperature,
        seed=ns.seed,
        cond_velocity=ns.cond_velocity,
        cond_kick=ns.cond_kick,
    )
    import json
    import sys

    if ns.out is None:
        json.dump(ev, sys.stdout)
    else:
        with ns.out.open("w") as fh:
            json.dump(ev, fh)


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: modcompose <command> [<args>]\n\ncommands: demo, sample")
        sys.exit(0)
    cmd, *rest = argv
    if cmd == "demo":
        _cmd_demo(rest)
    elif cmd == "sample":
        _cmd_sample(rest)
    else:
        sys.exit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
