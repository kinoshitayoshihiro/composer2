from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import re

import torch
from torch import nn

from models.phrase_transformer import PhraseTransformer


def _describe_instrument(inst) -> str:
    try:
        import pretty_midi as _pm
        prog = _pm.program_to_instrument_name(inst.program)
    except Exception:
        prog = str(getattr(inst, "program", ""))
    name = (inst.name or "").strip()
    return f"{name} {prog}".strip()


def _pick_instrument_idx(pm, inst_index: int | None, inst_regex: str | None, pitch_range: tuple[int, int] | None) -> int:
    n = len(pm.instruments)
    # 1) explicit index
    if inst_index is not None and 0 <= inst_index < n and pm.instruments[inst_index].notes:
        return inst_index
    # 2) regex on name/program
    if inst_regex:
        pat = re.compile(inst_regex, re.I)
        for i, inst in enumerate(pm.instruments):
            desc = _describe_instrument(inst)
            if pat.search(desc) and inst.notes:
                return i
    # 3) pitch range vote: pick instrument with most notes in range
    if pitch_range is not None:
        lo, hi = pitch_range
        best_i, best_c = 0, -1
        for i, inst in enumerate(pm.instruments):
            c = sum(1 for n in inst.notes if lo <= n.pitch <= hi)
            if c > best_c:
                best_c, best_i = c, i
        if best_c > 0:
            return best_i
    # 4) fallback to first non-empty
    for i, inst in enumerate(pm.instruments):
        if inst.notes:
            return i
    return 0


def _midi_to_feats(
    data: bytes,
    *,
    inst_index: int | None = None,
    inst_regex: str | None = None,
    pitch_range: tuple[int, int] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[float]]:
    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
        idx = _pick_instrument_idx(pm, inst_index, inst_regex, pitch_range)
        inst = pm.instruments[idx]
        notes = inst.notes
        if pitch_range is not None:
            lo, hi = pitch_range
            notes = [n for n in notes if lo <= n.pitch <= hi]
        notes = sorted(notes, key=lambda n: n.start)
    except Exception:
        try:
            import miditoolkit
        except Exception as exc:
            raise ImportError(
                "pretty_midi or miditoolkit required for MIDI parsing"
            ) from exc
        mt = miditoolkit.MidiFile(file=io.BytesIO(data))
        # limited support: ignore selection for miditoolkit path; use first instrument
        mt_notes = mt.instruments[0].notes
        notes = sorted(mt_notes, key=lambda n: n.start)
    pc = torch.tensor([n.pitch % 12 for n in notes], dtype=torch.long).unsqueeze(0)
    vel = torch.tensor([n.velocity for n in notes], dtype=torch.float32).unsqueeze(0)
    dur = torch.tensor([n.end - n.start for n in notes], dtype=torch.float32).unsqueeze(
        0
    )
    pos = torch.arange(len(notes), dtype=torch.long).unsqueeze(0)
    mask = torch.ones(1, len(notes), dtype=torch.bool)
    feats = {"pitch_class": pc, "velocity": vel, "duration": dur, "position": pos}
    note_times = [float(n.start) for n in notes]
    return feats, mask, note_times


class PhraseLSTMInfer(nn.Module):  # minimal LSTM to load LSTM checkpoints
    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 256,
        *,
        duv_mode: str = "none",
        vel_bins: int = 0,
        dur_bins: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        self.feat_proj = nn.Linear(d_model, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True
        )
        self.head_boundary = nn.Linear(d_model, 1)
        # optional heads (not used during segmentation but allow checkpoint load)
        self.head_vel_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        self.head_dur_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        self.head_vel_cls = (
            nn.Linear(d_model, vel_bins) if duv_mode in {"cls", "both"} and vel_bins > 0 else None
        )
        self.head_dur_cls = (
            nn.Linear(d_model, dur_bins) if duv_mode in {"cls", "both"} and dur_bins > 0 else None
        )

    def forward(self, feats: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pos_ids = feats["position"].clamp(max=self.max_len - 1)
        dur = self.dur_proj(feats["duration"].unsqueeze(-1))
        vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
        pc = self.pitch_emb(feats["pitch_class"] % 12)
        pos = self.pos_emb(pos_ids)
        x = torch.cat([dur, vel, pc, pos], dim=-1)
        x = self.feat_proj(x)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        # Pad to the longest sequence in the batch (variable length inference)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=None
        )
        outputs: dict[str, torch.Tensor] = {}
        outputs["boundary"] = self.head_boundary(out).squeeze(-1)
        if self.head_vel_reg is not None:
            outputs["vel_reg"] = self.head_vel_reg(out).squeeze(-1)
        if self.head_dur_reg is not None:
            outputs["dur_reg"] = self.head_dur_reg(out).squeeze(-1)
        if self.head_vel_cls is not None:
            outputs["vel_cls"] = self.head_vel_cls(out)
        if self.head_dur_cls is not None:
            outputs["dur_cls"] = self.head_dur_cls(out)
        return outputs


def _build_from_meta(state: dict, default_arch: str = "transformer") -> nn.Module:
    meta = state.get("meta", {})
    arch = meta.get("arch", default_arch)
    d_model = int(meta.get("d_model", 256))
    max_len = int(meta.get("max_len", 256))
    duv_mode = str(meta.get("duv_mode", "none"))
    vel_bins = int(meta.get("vel_bins", 0))
    dur_bins = int(meta.get("dur_bins", 0))
    if arch == "lstm":
        return PhraseLSTMInfer(d_model=d_model, max_len=max_len, duv_mode=duv_mode, vel_bins=vel_bins, dur_bins=dur_bins)
    # transformer defaults
    n_layers = int(meta.get("n_layers", meta.get("n_layers", 4)))
    n_heads = int(meta.get("n_heads", 8))
    dropout = float(meta.get("dropout", 0.1))
    return PhraseTransformer(
        d_model=d_model,
        max_len=max_len,
        section_vocab_size=0,
        mood_vocab_size=0,
        vel_bucket_size=vel_bins if duv_mode in {"cls", "both"} else 0,
        dur_bucket_size=dur_bins if duv_mode in {"cls", "both"} else 0,
        duv_mode=duv_mode,
        vel_bins=vel_bins,
        dur_bins=dur_bins,
        nhead=n_heads,
        num_layers=n_layers,
        dropout=dropout,
    )


def load_model(arch: str, ckpt: Path) -> nn.Module:
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model = _build_from_meta(state, default_arch=arch)
    # tolerate missing/unexpected keys when shapes align
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model


def segment_bytes(
    data: bytes,
    model: nn.Module,
    threshold: float,
    *,
    inst_index: int | None = None,
    inst_regex: str | None = None,
    pitch_range: tuple[int, int] | None = None,
) -> list[tuple[int, float]]:
    try:
        feats, mask, _ = _midi_to_feats(
            data,
            inst_index=inst_index,
            inst_regex=inst_regex,
            pitch_range=pitch_range,
        )
    except TypeError:
        feats, mask, _ = _midi_to_feats(data)
    with torch.no_grad():
        outputs = model(feats, mask)
        if "boundary" not in outputs:
            logging.warning("segmenter outputs missing 'boundary' logits")
        logits = outputs["boundary"][0]
        valid = mask[0].bool()
        probs = torch.sigmoid(logits[valid])
    return [(int(i), float(p)) for i, p in enumerate(probs.tolist()) if p > threshold]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("--ckpt", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument(
        "--arch", choices=["transformer", "lstm"], default="transformer"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--instrument-index", type=int, default=None)
    parser.add_argument("--instrument-regex", type=str, default=None)
    parser.add_argument("--pitch-range", type=int, nargs=2, metavar=("LOW","HIGH"), default=None)
    parser.add_argument("--out-tsv", type=Path, default=None, help="write TSV (index,prob,seconds,beats)")
    args = parser.parse_args(argv)
    data = args.midi.read_bytes()
    model = load_model(args.arch, args.ckpt)
    try:
        feats, mask, note_times = _midi_to_feats(
            data,
            inst_index=args.instrument_index,
            inst_regex=args.instrument_regex,
            pitch_range=tuple(args.pitch_range) if args.pitch_range else None,
        )
    except TypeError:
        feats, mask, note_times = _midi_to_feats(data)
    with torch.no_grad():
        outputs = model(feats, mask)
        if "boundary" not in outputs:
            logging.warning("segmenter outputs missing 'boundary' logits")
        logits = outputs["boundary"][0]
        valid = mask[0].bool()
        probs = torch.sigmoid(logits[valid]).cpu().tolist()
    rows = [(i, float(p)) for i, p in enumerate(probs) if p > args.threshold]
    if args.out_tsv:
        try:
            import pretty_midi
            from bisect import bisect_right
            pm = pretty_midi.PrettyMIDI(args.midi.as_posix())
            tpb = float(pm.resolution)
            beat_times = pm.get_beats()  # ascending times of beats
            downbeats = pm.get_downbeats()  # bar starts
            # map downbeat time -> nearest beat index <= t
            def beat_index_at(t: float) -> int:
                if not beat_times:
                    return 0
                i = max(0, bisect_right(beat_times, t) - 1)
                return min(i, len(beat_times) - 1)

            downbeat_beat_idx = [beat_index_at(db) for db in downbeats] if downbeats is not None else []

            def global_beat_float(t: float) -> float:
                if not beat_times:
                    return 0.0
                i = beat_index_at(t)
                if i >= len(beat_times) - 1:
                    return float(i)
                dt = beat_times[i + 1] - beat_times[i]
                frac = 0.0 if dt <= 0 else (t - beat_times[i]) / dt
                return float(i) + float(max(0.0, min(1.0, frac)))

            def bar_beat_at(t: float) -> tuple[int, int, float, int]:
                # bar: 1-based index of last downbeat <= t
                if downbeats:
                    b = max(0, bisect_right(downbeats, t) - 1)
                    bar = b + 1
                    b0_idx = downbeat_beat_idx[b] if b < len(downbeat_beat_idx) else 0
                else:
                    bar, b0_idx = 1, 0
                i = beat_index_at(t)
                # beat number in bar is 1-based
                beat_num = int(max(0, i - b0_idx)) + 1
                gb = global_beat_float(t)
                tick = int(round(gb * tpb))
                # fractional beat within bar (global minus bar start beats)
                bf = gb - float(b0_idx)
                return bar, beat_num, bf, tick

        except Exception:
            beat_times = []
            def bar_beat_at(t: float) -> tuple[int, int, float, int]:
                return 1, 1, 0.0, 0

        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w") as f:
            f.write("idx\tprob\tsec\tbar\tbeat\tbeat_float\ttick\n")
            for i, p in rows:
                t = note_times[i] if i < len(note_times) else 0.0
                bar, beat, bf, tick = bar_beat_at(t)
                f.write(f"{i}\t{p:.3f}\t{t:.3f}\t{bar}\t{beat}\t{bf:.2f}\t{tick}\n")
    else:
        for b in rows:
            print(b)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
