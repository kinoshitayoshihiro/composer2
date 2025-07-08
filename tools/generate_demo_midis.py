import argparse
import subprocess
from pathlib import Path
import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: str, sections: list[str] | None = None) -> None:
    cfg = load_cfg(cfg_path)
    target_sections = sections or cfg.get("sections_to_generate", [])
    if not target_sections and "Sax Solo" in cfg.get("sections_to_generate", []):
        target_sections = ["Sax Solo"]
    for section in target_sections:
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
    parser.add_argument("--sections", nargs="*", help="Override sections to generate")
    args = parser.parse_args()
    Path("demos").mkdir(exist_ok=True)
    main(args.main_cfg, args.sections)
