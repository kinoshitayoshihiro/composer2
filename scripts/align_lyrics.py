from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import pretty_midi
import torch

from models.lyrics_alignment import LyricsAligner


def load_model(
    path: Path, device: str = "cpu"
) -> tuple[LyricsAligner, dict, list[str]]:
    state = torch.load(path, map_location=device)
    cfg = state["config"]
    vocab = state["vocab"]
    model = LyricsAligner(
        len(vocab),
        cfg["midi_feature_dim"],
        cfg["hidden_size"],
        cfg["dropout"],
        cfg["ctc_blank"],
        freeze_encoder=cfg.get("freeze_encoder", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)
    return model, cfg, vocab


def _midi_frames(times: list[float], n_frames: int, hop_length_ms: int) -> torch.Tensor:
    """Map MIDI onset times in milliseconds to frame indices."""
    arr = torch.zeros(n_frames, dtype=torch.long)
    for t_ms in times:
        idx = int(t_ms / hop_length_ms)
        if idx < n_frames:
            # インデックスを 0..511 に制限
            arr[idx] = min(idx, 511)
    return arr


def align_audio(
    audio: torch.Tensor,
    midi_times: list[float],
    model: LyricsAligner,
    cfg: dict,
    vocab: list[str],
):
    n_frames = int(
        round(len(audio) / cfg["sample_rate"] / (cfg["hop_length_ms"] / 1000))
    )
    if n_frames == 0 or not midi_times:
        return []
    midi = _midi_frames(midi_times, n_frames, cfg["hop_length_ms"]).unsqueeze(0)
    with torch.no_grad():
        logp = model(audio.unsqueeze(0), midi)
    pred = logp.argmax(-1)[:n_frames, 0]  # (T,B) -> use first batch
    res = []
    prev = len(vocab)
    for i, p in enumerate(pred):
        idx = int(p)
        if idx != prev and idx != len(vocab):
            res.append({"phoneme": vocab[idx], "time_ms": i * cfg["hop_length_ms"]})
        prev = idx
    return res


def align_pair(
    audio: Path, midi: Path, model: LyricsAligner, cfg: dict, vocab: list[str]
):
    wav, _ = librosa.load(audio, sr=cfg["sample_rate"])
    pm = pretty_midi.PrettyMIDI(str(midi))
    times = [n.start * 1000 for n in pm.instruments[0].notes]
    return align_audio(torch.tensor(wav, dtype=torch.float32), times, model, cfg, vocab)


# `align_file` kept for backward compatibility in tests
align_file = align_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--midi", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.batch:
        items = json.loads(Path(args.batch).read_text())
    else:
        if args.audio is None or args.midi is None:
            print("missing file", file=sys.stderr)
            return 1
        items = [{"audio": str(args.audio), "midi": str(args.midi)}]
    for it in items:
        if not Path(it["audio"]).exists() or not Path(it["midi"]).exists():
            print("missing file", file=sys.stderr)
            return 1
    if not args.ckpt.exists():
        print("missing file", file=sys.stderr)
        return 1
    model, cfg, vocab = load_model(args.ckpt, args.device)
    results = [
        align_pair(Path(d["audio"]), Path(d["midi"]), model, cfg, vocab) for d in items
    ]
    out_str = json.dumps(results)
    if args.out_json:
        args.out_json.write_text(out_str)
    else:
        print(out_str)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
