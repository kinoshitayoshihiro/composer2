from __future__ import annotations

from pathlib import Path

from tools.dump_tree import build_tree, render_markdown


def main(root: str | Path) -> Path:
    base = Path(root).resolve()
    node, _ = build_tree(base)
    text = render_markdown(node)
    out = base / "tree.md"
    out.write_text(text)
    return out


if __name__ == "__main__":
    print(main(Path.cwd()))
