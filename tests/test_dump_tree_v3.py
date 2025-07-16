from pathlib import Path

from click.testing import CliRunner

from modular_composer.cli import cli


def test_dump_tree_v3(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("x")
    runner = CliRunner()
    res = runner.invoke(cli, ["dump-tree", str(root), "--version", "3"])
    assert res.exit_code == 0
    out = root / "tree.md"
    assert out.exists()
    text = out.read_text()
    assert "# Project Tree v3" in text
    assert "```" in text
    arch_lines = Path("docs/ARCH.md").read_text().splitlines()
    snippet = arch_lines[14].strip()
    assert snippet == "# Project Tree v3"
    assert snippet in text
