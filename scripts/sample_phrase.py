from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pretty_midi
import numpy as np
import torch

from models.phrase_transformer import PhraseTransformer
from utilities.phrase_data import denorm_duv


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load note rows from CSV or JSON file."""
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            return [{k: float(v) if k in {"velocity", "duration"} else int(v) for k, v in row.items() if v != ""} for row in reader]
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open() as f:
            data = json.load(f)
            if isinstance(data, list):
                return data  # type: ignore[return-value]
    raise SystemExit("unsupported input format")


def _build_feats(rows: List[Dict[str, Any]], max_len: int) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Convert rows to model feature tensors and mask."""
    length = min(len(rows), max_len)
    mask = torch.zeros(1, max_len, dtype=torch.bool)
    mask[0, :length] = 1
    pad = {"pitch_class": 0, "position": 0, "duration": 0.0, "velocity": 0.0}
    feats: Dict[str, torch.Tensor] = {}
    pc = [int(rows[i].get("pitch", 0)) % 12 if i < length else pad["pitch_class"] for i in range(max_len)]
    pos = [int(rows[i].get("pos", 0)) if i < length else pad["position"] for i in range(max_len)]
    dur = [float(rows[i].get("duration", 0.0)) if i < length else pad["duration"] for i in range(max_len)]
    vel = [float(rows[i].get("velocity", 0.0)) if i < length else pad["velocity"] for i in range(max_len)]
    feats["pitch_class"] = torch.tensor(pc, dtype=torch.long).unsqueeze(0)
    feats["position"] = torch.tensor(pos, dtype=torch.long).unsqueeze(0)
    feats["duration"] = torch.tensor(dur, dtype=torch.float32).unsqueeze(0)
    feats["velocity"] = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    if "velocity_bucket" in rows[0]:
        vb = [int(rows[i].get("velocity_bucket", 0)) if i < length else 0 for i in range(max_len)]
        feats["vel_bucket"] = torch.tensor(vb, dtype=torch.long).unsqueeze(0)
    if "duration_bucket" in rows[0]:
        db = [int(rows[i].get("duration_bucket", 0)) if i < length else 0 for i in range(max_len)]
        feats["dur_bucket"] = torch.tensor(db, dtype=torch.long).unsqueeze(0)
    return feats, mask


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--in", "--infile", dest="inp", type=Path, required=True)
    parser.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--duv-mode", choices=["none", "reg", "cls", "both"], default="reg")
    parser.add_argument("--out-midi", type=Path, required=True)
    parser.add_argument("--tempo", type=float, default=None, help="MIDI tempo in BPM; defaults to estimated tempo")
    parser.add_argument("--ppq", type=int, default=480, help="ticks per quarter note")
    parser.add_argument("--ts", type=str, default="4/4", help="time signature, e.g. 3/4")
    parser.add_argument("--program", type=int, default=0, help="MIDI program number for instrument")
    args = parser.parse_args(argv)

    state = torch.load(args.ckpt, map_location="cpu")
    duv_cfg = state.get("duv_cfg", {})
    vel_bins = duv_cfg.get("vel_bins", 0)
    dur_bins = duv_cfg.get("dur_bins", 0)

    if args.arch == "lstm":
        from scripts.train_phrase import PhraseLSTM  # type: ignore

        model = PhraseLSTM(
            d_model=state.get("d_model", 128),
            max_len=args.max_len,
            duv_mode=args.duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
        )
    else:
        model = PhraseTransformer(
            d_model=state.get("d_model", 512),
            max_len=args.max_len,
            duv_mode=args.duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
        )
    model.load_state_dict(state["model"])
    model.eval()

    rows = _load_rows(args.inp)
    feats, mask = _build_feats(rows, args.max_len)
    with torch.no_grad():
        outputs = model({k: v for k, v in feats.items()}, mask)
    vel_reg = outputs.get("vel_reg")
    dur_reg = outputs.get("dur_reg")
    if vel_reg is None or dur_reg is None:
        raise SystemExit("velocity/duration regression required for MIDI output")
    vel, dur = denorm_duv(vel_reg[0], dur_reg[0])

    pm = pretty_midi.PrettyMIDI(resolution=args.ppq, initial_tempo=args.tempo or 120.0)
    inst = pretty_midi.Instrument(program=args.program)
    t = 0.0
    for i, row in enumerate(rows[: len(vel)]):
        pitch = int(row.get("pitch", 60))
        v = int(vel[i].item())
        d = float(dur[i].item())
        inst.notes.append(pretty_midi.Note(pitch=pitch, velocity=v, start=t, end=t + d))
        t += d
    pm.instruments.append(inst)
    num, denom = map(int, args.ts.split("/"))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))
    if args.tempo is None:
        tempo = float(pm.estimate_tempo())
        pm._tempo_changes = (np.array([0.0]), np.array([tempo]))
        if hasattr(pm, "_update_tick_to_time"):
            pm._update_tick_to_time()
    pm.write(str(args.out_midi))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
