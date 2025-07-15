import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml


def normalize_section(name: str) -> str:
    """Normalize section name for matching."""
    return " ".join(name.replace("_", " ").replace("\u2011", "-").strip().split())


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: str, sections: list[str] | None = None) -> None:
    model_path = Path("models/groove_ngram.pkl")
    if not model_path.exists():
        print("\u26a0 Model not found; continuing without generation", file=sys.stderr)

    cfg = load_cfg(cfg_path)
    available = {normalize_section(s): s for s in cfg.get("sections_to_generate", [])}

    target_sections = sections or list(available.keys())
    if not target_sections and "Sax Solo" in available:
        target_sections = ["Sax Solo"]

    for section in target_sections:
        normalized = normalize_section(section)
        if normalized not in available:
            print(f"[warn] unknown section: {section}", file=sys.stderr)
            continue
        out_name = f"demo_{normalized.replace(' ', '_')}.mid"
        try:
            cmd = [
                "python3",
                "modular_composer.py",
                "-m",
                cfg_path,
                "--dry-run",
                "--output-dir",
                "demos",
                "--output-filename",
                out_name,
            ]
            drums_cond = cfg.get("part_defaults", {}).get("drums", {}).get("cond", {})
            if isinstance(drums_cond, dict):
                if drums_cond.get("style"):
                    cmd += ["--cond-style", drums_cond["style"]]
                if drums_cond.get("feel"):
                    cmd += ["--cond-feel", drums_cond["feel"]]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"failed to generate {section}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo MIDIs per section")
    parser.add_argument("-m", "--main-cfg", required=True)
    parser.add_argument("--sections", nargs="*", help="Override sections to generate")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        print("\u26a0 No display backend; skipping demo generation", file=sys.stderr)
        sys.exit(0)
    sf2 = os.getenv("MC_SF2")
    if not sf2 and not Path("assets/default.sf2").exists():
        print("\u26a0 No SoundFont found; skipping audio render", file=sys.stderr)
        sys.exit(0)

    Path("demos").mkdir(exist_ok=True)
    main(args.main_cfg, args.sections)
