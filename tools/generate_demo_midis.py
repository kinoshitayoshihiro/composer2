import argparse
import subprocess
from pathlib import Path
import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    for section in cfg.get("sections_to_generate", []):
        out_name = f"demo_{section.replace(' ', '_')}.mid"
        subprocess.run(
            [
                "python3",
                "modular_composer.py",
                "-m",
                cfg_path,
                "--dry-run",
                "--output-dir",
                "demos",
                "--output-filename",
                out_name,
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo MIDIs per section")
    parser.add_argument("-m", "--main-cfg", required=True)
    args = parser.parse_args()
    Path("demos").mkdir(exist_ok=True)
    main(args.main_cfg)
