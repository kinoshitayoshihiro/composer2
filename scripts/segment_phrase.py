from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from torch import nn

from models.phrase_transformer import PhraseTransformer
from scripts.train_phrase import PhraseLSTM


def _midi_to_feats(data: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    try:
        import pretty_midi

        pm_notes = pretty_midi.PrettyMIDI(io.BytesIO(data)).instruments[0].notes
        notes = sorted(pm_notes, key=lambda n: n.start)
    except Exception:
        try:
            import miditoolkit
        except Exception as exc:
            raise ImportError(
                "pretty_midi or miditoolkit required for MIDI parsing"
            ) from exc
        mt_notes = miditoolkit.MidiFile(file=io.BytesIO(data)).instruments[0].notes
        notes = sorted(mt_notes, key=lambda n: n.start)
    pc = torch.tensor([n.pitch % 12 for n in notes], dtype=torch.long).unsqueeze(0)
    vel = torch.tensor([n.velocity for n in notes], dtype=torch.float32).unsqueeze(0)
    dur = torch.tensor([n.end - n.start for n in notes], dtype=torch.float32).unsqueeze(
        0
    )
    pos = torch.arange(len(notes), dtype=torch.long).unsqueeze(0)
    mask = torch.ones(1, len(notes), dtype=torch.bool)
    feats = {"pitch_class": pc, "velocity": vel, "duration": dur, "position": pos}
    return feats, mask


def load_model(arch: str, ckpt: Path) -> nn.Module:
    if arch == "lstm":
        model: nn.Module = PhraseLSTM()
    else:
        model = PhraseTransformer()
    if ckpt.is_file():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def segment_bytes(
    data: bytes, model: nn.Module, threshold: float
) -> list[tuple[int, float]]:
    feats, mask = _midi_to_feats(data)
    with torch.no_grad():
        logits = model(feats, mask)[0]
        probs = torch.sigmoid(logits)
    return [(int(i), float(p)) for i, p in enumerate(probs.tolist()) if p > threshold]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("--ckpt", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument(
        "--arch", choices=["transformer", "lstm"], default="transformer"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    data = args.midi.read_bytes()
    model = load_model(args.arch, args.ckpt)
    boundaries = segment_bytes(data, model, args.threshold)
    for b in boundaries:
        print(b)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
