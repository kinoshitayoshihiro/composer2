from pathlib import Path

from modular_composer import cli


def test_dump_tree_v3(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("x")
    cli.main(["dump-tree", str(root), "--version", "3"])
    out = root / "tree.md"
    assert out.exists()
    text = out.read_text()
    assert text.startswith("```")
