from __future__ import annotations

import argparse
from pathlib import Path

from tools.dump_tree import build_tree, render_markdown


def main(root: Path, version: int = 3) -> Path:
    base = root.resolve()
    node, _ = build_tree(base)
    tree = render_markdown(node or {})
    text = "# Project Tree v3\n\n```\n" + tree.strip() + "\n```\n"
    out = base / "tree.md"
    out.write_text(text)
    return out




def _argparse_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--version", type=int, default=3)
    ns = ap.parse_args(argv)
    path = main(ns.root, ns.version)
    print(path)


if __name__ == "__main__":
    _argparse_main()
