from pathlib import Path
import yaml
import subprocess

from tools.generate_demo_midis import main as generate


def test_generate_demo(tmp_path, monkeypatch):
    cfg = {"sections_to_generate": ["Verse 1"]}
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    def fake_run(cmd, check=True):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        fname = cmd[cmd.index("--output-filename") + 1]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / fname).touch()
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    generate(str(cfg_path))
    assert Path("demos/demo_Verse_1.mid").exists()
