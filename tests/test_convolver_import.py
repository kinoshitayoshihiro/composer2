import sys
import subprocess
from pathlib import Path


def test_editable_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    python = env_dir / "bin" / "python"
    pip = env_dir / "bin" / "pip"
    subprocess.run([pip, "install", "--no-deps", "-e", str(root)], check=True)
    subprocess.run([python, "-c", "import utilities.convolver"], check=True)
