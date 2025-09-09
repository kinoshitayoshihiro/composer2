import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml
import pretty_midi

from utilities.time_utils import get_end_time

from modular_composer.perc_generator import PercGenerator
from generator.drum_generator import DrumGenerator
from utilities.perc_sampler_v1 import PERC_NOTE_MAP


def normalize_section(name: str) -> str:
    """Normalize section name for matching."""
    return " ".join(name.replace("_", " ").replace("\u2011", "-").strip().split())


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


LABEL_TO_PITCH = {v: k for k, v in PERC_NOTE_MAP.items()}


def add_percussion(mid_path: Path, cfg: dict) -> None:
    """Append percussion events to *mid_path* according to *cfg*."""
    perc_cfg = cfg.get("part_defaults", {}).get("percussion")
    if not perc_cfg:
        return
    model_path = perc_cfg.get("model_path")
    if not model_path:
        return
    gen = PercGenerator(model_path)
    if gen.model is None:
        return

    ts = cfg.get("global_settings", {}).get("time_signature", "4/4")
    beats_per_bar = int(ts.split("/")[0])
    tempo = float(cfg.get("global_settings", {}).get("tempo_bpm", 120))
    bar_dur = 60.0 / tempo * beats_per_bar

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    num_bars = int(get_end_time(pm) / bar_dur + 0.999)
    perc_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Percussion")

    for bar in range(num_bars):
        start = bar * bar_dur
        drum_events = []
        for inst in pm.instruments:
            if not inst.is_drum:
                continue
            for n in inst.notes:
                if start <= n.start < start + bar_dur:
                    off = (n.start - start) / bar_dur
                    if n.pitch in {35, 36}:
                        drum_events.append({"instrument": "kick", "offset": off})
                    elif n.pitch in {38, 40}:
                        drum_events.append({"instrument": "snare", "offset": off})
        events = gen.generate_bar()
        merged = DrumGenerator.merge_perc_events(drum_events, events)
        for ev in merged:
            if ev.get("instrument") in {"kick", "snare"}:
                continue
            pitch = LABEL_TO_PITCH.get(ev["instrument"], 39)
            ev_start = start + ev["offset"] * bar_dur
            duration = ev.get("duration", 1 / gen.model.resolution) * bar_dur
            velocity = ev.get("velocity", 80)
            perc_inst.notes.append(
                pretty_midi.Note(velocity, pitch, ev_start, ev_start + duration)
            )

    if perc_inst.notes:
        pm.instruments.append(perc_inst)
        pm.write(str(mid_path))


def main(cfg_path: str, sections: list[str] | None = None) -> None:
    model_path = Path("models/groove_ngram.pkl")
    perc_path = Path("models/perc_ngram.pkl")
    if not model_path.exists() and not perc_path.exists():
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
        else:
            out_file = Path("demos") / out_name
            add_percussion(out_file, cfg)


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
