from __future__ import annotations

"""Predict phrase boundaries from a trained checkpoint on CSV or MIDI.

Usage (CSV):
  python -m scripts.predict_phrase \
      --ckpt checkpoints/guitar_lead/gtr_lead_fix_group.ckpt \
      --in-csv data/phrase_csv/gtr_lead_valid.csv \
      --out-csv checkpoints/guitar_lead/preds/gtr_lead_valid_preds.csv \
      --th 0.51 --device mps

Usage (MIDI):
  python -m scripts.predict_phrase \
      --ckpt checkpoints/guitar_lead/gtr_lead_fix_group.ckpt \
      --in-midi data/songs \
      --out-csv checkpoints/guitar_lead/preds/songs_preds.csv \
      --instrument-regex '(?i)guitar|DL_Guitar|ギター' --pitch-range 52 88 \
      --th 0.51 --device mps

Optional: threshold auto-calibration on a labeled CSV (e.g., validation):
  --val-csv data/phrase_csv/gtr_lead_valid.csv --scan-range 0.30 0.95 0.01 --auto-th

The script reconstructs the model from checkpoint meta, runs inference on
the provided rows, and writes a CSV with pred_prob and pred_boundary
columns appended. When --val-csv is provided with --auto-th, the best
threshold is selected by F1 scan and used for inference.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="trained checkpoint (.ckpt)")
    p.add_argument("--in-csv", type=Path, help="input phrase CSV")
    p.add_argument("--in-midi", type=Path, help="input MIDI file or directory (recursively)")
    p.add_argument("--out-csv", type=Path, required=True, help="output CSV with predictions")
    p.add_argument("--th", type=float, default=None, help="decision threshold for boundary (overridden by --auto-th if provided)")
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="inference device",
    )
    p.add_argument("--max-len", type=int, default=256, help="max sequence length")
    p.add_argument("--progress", action="store_true")
    # MIDI parsing options
    p.add_argument("--instrument", type=str)
    p.add_argument("--instrument-regex", type=str)
    p.add_argument("--pitch-range", type=int, nargs=2, metavar=("LOW", "HIGH"))
    p.add_argument("--boundary-gap-beats", type=float, default=1.05)
    p.add_argument("--boundary-on-section-change", action="store_true")
    p.add_argument("--no-path-hint", action="store_true", help="disable path-based instrument hint when parsing MIDI")
    # Threshold auto-calibration on labeled CSV
    p.add_argument("--val-csv", type=Path, help="validation CSV with ground-truth boundary")
    p.add_argument("--scan-range", type=float, nargs=3, default=(0.3, 0.95, 0.01), metavar=("START", "END", "STEP"))
    p.add_argument("--auto-th", action="store_true", help="calibrate threshold on --val-csv and use it for inference")
    # Reporting
    p.add_argument("--report-json", type=Path, help="write metrics/threshold summary JSON here")
    p.add_argument("--viz", action="store_true", help="save PR/CM plots when ground-truth is available")
    p.add_argument("--viz-dir", type=Path, help="directory to save visualizations (defaults to out_csv parent)")
    p.add_argument("--viz-sort", choices=["len", "f1_asc", "f1_desc"], default="len", help="ordering for per-song visualizations when limiting count")
    p.add_argument("--viz-pdf", type=Path, help="write a multi-page PDF with per-song plots (PR/CM/timeseries) for selected songs")
    # Append mode for large batch runs (useful when iterating files)
    p.add_argument("--append", action="store_true", help="append to --out-csv if it exists (write header only when creating)")
    # Song ID handling
    p.add_argument("--song-id-col", type=str, help="column name in CSV that holds song ID")
    p.add_argument(
        "--derive-song-id-from",
        type=str,
        help="column name to derive song_id from (uses stem of path)",
    )
    p.add_argument("--require-song-id", action="store_true", help="fail if song_id cannot be obtained from CSV columns")
    # Per-song adaptive threshold
    p.add_argument(
        "--per-song-th",
        action="store_true",
        help="use per-song best threshold when ground-truth is available (input CSV) or from --val-csv mapping",
    )
    p.add_argument(
        "--viz-timeseries",
        type=int,
        default=0,
        help="save probability timeseries per song (limit to N songs; 0 to disable)",
    )
    # Post-processing
    p.add_argument("--th-on", type=float, help="hysteresis ON threshold (defaults to --th)")
    p.add_argument("--th-off", type=float, help="hysteresis OFF threshold (defaults to --th)")
    p.add_argument("--min-gap", type=int, default=1, help="minimum gap between boundaries (in steps)")
    return p


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def write_csv(rows: Iterable[dict[str, object]], path: Path, fieldnames: List[str], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    need_header = True
    if append and path.exists() and path.stat().st_size > 0:
        need_header = False
    with path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()
        w.writerows(rows)


def setup_device(device_str: str):
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
        elif torch.backends.mps.is_available():
            d = torch.device("mps")
        else:
            d = torch.device("cpu")
    else:
        d = torch.device(device_str)
    if d.type == "mps":
        torch.set_float32_matmul_precision("medium")
    return d


def reconstruct_model(ckpt_path: Path, device):
    import torch
    from torch import nn

    state = torch.load(ckpt_path, map_location="cpu")
    meta = state.get("meta", {})
    arch = meta.get("arch", "lstm")
    d_model = int(meta.get("d_model", 256))
    max_len = int(meta.get("max_len", 256))
    duv_mode = str(meta.get("duv_mode", "reg"))
    vel_bins = int(meta.get("vel_bins", 0))
    dur_bins = int(meta.get("dur_bins", 0))
    n_layers = int(meta.get("n_layers", 4))
    n_heads = int(meta.get("n_heads", 8))

    # Determine embedding table sizes from checkpoint (preferred) to avoid mismatches
    sd_full = state.get("model", state)
    vb_size = 0
    db_size = 0
    feat_in_ckpt = None
    if isinstance(sd_full, dict):
        w = sd_full.get("vel_bucket_emb.weight")
        if w is not None:
            vb_size = int(w.shape[0])
        w = sd_full.get("dur_bucket_emb.weight")
        if w is not None:
            db_size = int(w.shape[0])
        w = sd_full.get("feat_proj.weight")
        if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
            # torch Linear weight shape: (out_features, in_features)
            feat_in_ckpt = int(w.shape[1])

    class PhraseLSTM(nn.Module):  # minimal replica for inference (optionally with TCN/2-class head)
        def __init__(
            self,
            d_model: int,
            max_len: int,
            duv_mode: str,
            vel_bins: int,
            dur_bins: int,
            vel_bucket_size: int,
            dur_bucket_size: int,
            use_bar_beat: bool,
            use_tcn: bool = False,
            two_class_head: bool = False,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.pitch_emb = nn.Embedding(12, d_model // 4)
            self.pos_emb = nn.Embedding(max_len, d_model // 4)
            self.dur_proj = nn.Linear(1, d_model // 4)
            self.vel_proj = nn.Linear(1, d_model // 4)
            self.use_bar_beat = use_bar_beat
            if use_bar_beat:
                self.barpos_proj = nn.Linear(1, d_model // 8)
                self.beatpos_proj = nn.Linear(1, d_model // 8)
                extra_bar_beat = d_model // 4
            else:
                self.barpos_proj = None
                self.beatpos_proj = None
                extra_bar_beat = 0
            extra = 0
            self.vel_bucket_emb = None
            self.dur_bucket_emb = None
            # Use embedding sizes from checkpoint to match state_dict
            if vel_bucket_size > 0:
                self.vel_bucket_emb = nn.Embedding(vel_bucket_size, 8)
                extra += 8
            if dur_bucket_size > 0:
                self.dur_bucket_emb = nn.Embedding(dur_bucket_size, 8)
                extra += 8
            self.feat_proj = nn.Linear(d_model + extra + extra_bar_beat, d_model)
            self.lstm = nn.LSTM(
                d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True
            )
            self.use_tcn = bool(use_tcn)
            if self.use_tcn:
                self.tcn = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            else:
                self.tcn = None
            # boundary head(s)
            self.two_class_head = bool(two_class_head)
            if self.two_class_head:
                self.head_boundary2 = nn.Linear(d_model, 2)
                self.head_boundary = None
            else:
                self.head_boundary = nn.Linear(d_model, 1)
                self.head_boundary2 = None
            # heads below are created to match state_dict keys, though unused here
            self.head_vel_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
            self.head_dur_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
            self.head_vel_cls = nn.Linear(d_model, vel_bins) if duv_mode in {"cls", "both"} and vel_bins > 0 else None
            self.head_dur_cls = nn.Linear(d_model, dur_bins) if duv_mode in {"cls", "both"} and dur_bins > 0 else None

        def forward(self, feats, mask):
            import torch

            pos_ids = feats["position"].clamp(max=self.max_len - 1)
            dur = self.dur_proj(feats["duration"].unsqueeze(-1))
            vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
            pc = self.pitch_emb(feats["pitch_class"] % 12)
            pos = self.pos_emb(pos_ids)
            parts = [dur, vel, pc, pos]
            if self.use_bar_beat and "bar_phase" in feats and "beat_phase" in feats and self.barpos_proj is not None and self.beatpos_proj is not None:
                bp = self.barpos_proj(feats["bar_phase"].unsqueeze(-1))
                bt = self.beatpos_proj(feats["beat_phase"].unsqueeze(-1))
                parts.extend([bp, bt])
            if self.vel_bucket_emb is not None and "vel_bucket" in feats:
                parts.append(self.vel_bucket_emb(feats["vel_bucket"]))
            if self.dur_bucket_emb is not None and "dur_bucket" in feats:
                parts.append(self.dur_bucket_emb(feats["dur_bucket"]))
            x = torch.cat(parts, dim=-1)
            x = self.feat_proj(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=self.max_len
            )
            if self.tcn is not None:
                tmp = out.transpose(1, 2)
                tmp = self.tcn(tmp)
                out = tmp.transpose(1, 2)
            if self.two_class_head and self.head_boundary2 is not None:
                logits2 = self.head_boundary2(out)
                # use difference as single-logit boundary score
                logits = (logits2[..., 1] - logits2[..., 0]).squeeze(-1)
            elif self.head_boundary is not None:
                logits = self.head_boundary(out).squeeze(-1)
            else:
                logits = out.new_zeros(out.size(0), out.size(1))
            return {"boundary": logits}

    # Detect optional modules from state_dict
    has_two_class = isinstance(sd_full, dict) and any(k.startswith("head_boundary2.") for k in sd_full.keys())
    has_tcn = isinstance(sd_full, dict) and any(k.startswith("tcn.0.") for k in sd_full.keys())

    if arch in {"lstm", "bilstm_tcn"}:
        model = PhraseLSTM(
            d_model=d_model,
            max_len=max_len,
            duv_mode=duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
            vel_bucket_size=vb_size,
            dur_bucket_size=db_size,
            use_bar_beat=bool(meta.get("use_bar_beat", False)),
            use_tcn=has_tcn,
            two_class_head=has_two_class,
        )
    else:
        # Try to import transformer
        try:
            from models.phrase_transformer import PhraseTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemError("Transformer model not available; install models.phrase_transformer") from e
        # Prefer sizes detected from checkpoint to avoid mismatches
        vel_bucket_size = vb_size if vb_size > 0 else (vel_bins if duv_mode in {"cls", "both"} else 0)
        dur_bucket_size = db_size if db_size > 0 else (dur_bins if duv_mode in {"cls", "both"} else 0)
        # Infer bar/beat usage from checkpoint feat_proj input size if possible
        use_bar_beat_meta = bool(meta.get("use_bar_beat", False))
        use_bar_beat_sd = False
        if feat_in_ckpt is not None:
            extra_known = 0
            if vel_bucket_size:
                extra_known += 8
            if dur_bucket_size:
                extra_known += 8
            # If checkpoint's feat_proj expects +2*(d_model//8) beyond known extras, assume bar+beat
            if feat_in_ckpt == (d_model + extra_known + 2 * (d_model // 8)):
                use_bar_beat_sd = True
        use_bar_beat_tf = use_bar_beat_meta or use_bar_beat_sd
        model = PhraseTransformer(
            d_model=d_model,
            max_len=max_len,
            section_vocab_size=0,
            mood_vocab_size=0,
            vel_bucket_size=vel_bucket_size,
            dur_bucket_size=dur_bucket_size,
            use_bar_beat=use_bar_beat_tf,
            duv_mode=duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=0.1,
            use_sinusoidal_posenc=False,
        )

    sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    return model, meta


def build_groups(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    cur: list[dict[str, str]] = []
    prev_bar = None
    for r in rows:
        try:
            bar = int(r.get("bar", 0))
        except Exception:
            bar = 0
        if prev_bar is None:
            cur = [r]
            prev_bar = bar
            continue
        if bar != prev_bar:
            groups.append(sorted(cur, key=lambda x: int(x.get("pos", 0))))
            cur = [r]
            prev_bar = bar
        else:
            cur.append(r)
    if cur:
        groups.append(sorted(cur, key=lambda x: int(x.get("pos", 0))))
    return groups


def rows_to_feats(groups: list[list[dict[str, str]]], max_len: int, use_duv_embed: bool, use_bar_beat: bool, use_harmony: bool):
    import torch

    feats_list = []
    masks = []
    idx_maps = []  # map back to (group_index, row_index)
    for gi, g in enumerate(groups):
        L = min(len(g), max_len)
        pad = max_len - L
        pitches = [int(r.get("pitch", 0)) for r in g[:L]]
        pitch_cls = [p % 12 for p in pitches]
        vel = [float(r.get("velocity", 0)) for r in g[:L]]
        dur = [float(r.get("duration", 0)) for r in g[:L]]
        pos_vals = [int(r.get("pos", 0)) % max_len for r in g[:L]]
        pc_t = torch.tensor(pitch_cls + [0] * pad, dtype=torch.long)
        vel_t = torch.tensor(vel + [0] * pad, dtype=torch.float32)
        dur_t = torch.tensor(dur + [0] * pad, dtype=torch.float32)
        pos_t = torch.tensor(pos_vals + [0] * pad, dtype=torch.long)
        feats = {
            "pitch_class": pc_t,
            "velocity": vel_t,
            "duration": dur_t,
            "position": pos_t,
        }
        if use_bar_beat:
            L_eff = len(g[:L])
            bar_phase = [i / max(1, L_eff - 1) for i in range(L_eff)] + [0.0] * pad
            max_pos = float(max(pos_vals)) if pos_vals else 1.0
            beat_phase = [float(v) / max(1.0, max_pos) for v in pos_vals] + [0.0] * pad
            feats["bar_phase"] = torch.tensor(bar_phase, dtype=torch.float32)
            feats["beat_phase"] = torch.tensor(beat_phase, dtype=torch.float32)
        if use_duv_embed:
            vb = [int(r.get("velocity_bucket", 0)) for r in g[:L]] + [0] * pad
            db = [int(r.get("duration_bucket", 0)) for r in g[:L]] + [0] * pad
            feats["vel_bucket"] = torch.tensor(vb, dtype=torch.long)
            feats["dur_bucket"] = torch.tensor(db, dtype=torch.long)
        if use_harmony:
            hr = [int(r.get("harm_root", 0)) for r in g[:L]] + [0] * pad
            hf = [int(r.get("harm_func", 0)) for r in g[:L]] + [0] * pad
            hd = [int(r.get("harm_degree", 0)) for r in g[:L]] + [0] * pad
            feats["harm_root"] = torch.tensor(hr, dtype=torch.long)
            feats["harm_func"] = torch.tensor(hf, dtype=torch.long)
            feats["harm_degree"] = torch.tensor(hd, dtype=torch.long)
        feats_list.append(feats)
        m = torch.zeros(max_len, dtype=torch.bool)
        m[:L] = 1
        masks.append(m)
        idx_maps.append([(gi, i) for i in range(L)])
    return feats_list, masks, idx_maps


def predict(model, device, groups, max_len: int, use_duv_embed: bool, th: float, temp: float | None = None, use_bar_beat: bool = False, use_harmony: bool = False):
    import torch
    from torch.nn import functional as F

    feats_list, masks, idx_maps = rows_to_feats(groups, max_len=max_len, use_duv_embed=use_duv_embed, use_bar_beat=use_bar_beat, use_harmony=use_harmony)
    probs_flat: list[float] = []
    preds_flat: list[int] = []
    for feats, mask, idxs in zip(feats_list, masks, idx_maps):
        feats = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
        mask_b = mask.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feats, mask_b)
            logits = out["boundary"][0, mask]
            if temp and temp > 0:
                logits = logits / float(temp)
            probs = torch.sigmoid(logits).cpu().tolist()
        preds = [1 if p > th else 0 for p in probs]
        probs_flat.extend(probs)
        preds_flat.extend(preds)
    return probs_flat, preds_flat


def list_midi_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".mid", ".midi"}:
        return [root]
    return [p for p in root.rglob("*") if p.suffix.lower() in {".mid", ".midi"}]


def midi_to_rows_batch(
    paths: list[Path],
    boundary_gap_beats: float,
    instrument: str | None,
    instrument_regex: str | None,
    pitch_range: tuple[int, int] | None,
    *,
    path_hint: bool = True,
) -> list[dict[str, object]]:
    # Lazy import to avoid hard dependency
    from tools.corpus_to_phrase_csv import midi_to_rows  # type: ignore
    import re

    inst_re = re.compile(instrument_regex, re.I) if instrument_regex else None
    out: list[dict[str, object]] = []
    for p in paths:
        rows, _matched = midi_to_rows(
            p,
            boundary_gap_beats,
            sections=None,
            instrument_filter=instrument.lower() if instrument else None,
            instrument_regex=inst_re,
            pitch_range=pitch_range,
            path_hint_match=bool(path_hint),
            emit_buckets=True,  # ensure velocity/duration buckets available
            dur_bins=16,
            vel_bins=8,
        )
        # Attach song_id using stem; ensure stable ASCII-ish id if needed
        sid = p.stem
        for r in rows:
            r["song_id"] = sid
        out.extend(rows)
    return out


def f1_score(trues: list[int], preds: list[int]) -> float:
    tp = sum(1 for t, p in zip(trues, preds) if t and p)
    fp = sum(1 for t, p in zip(trues, preds) if not t and p)
    fn = sum(1 for t, p in zip(trues, preds) if t and not p)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def scan_thresholds(trues: list[int], probs: list[float], start: float, end: float, step: float) -> tuple[float, float, float | None]:
    # returns best_f1, best_th, pr_auc (if available)
    best_f1, best_th = -1.0, 0.5
    n = int(round((end - start) / step)) + 1
    ths = [round(start + i * step, 10) for i in range(n)]
    for th in ths:
        preds = [1 if p > th else 0 for p in probs]
        f1 = f1_score(trues, preds)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    try:
        from sklearn.metrics import average_precision_score  # type: ignore

        pr_auc = float(average_precision_score(trues, probs))
    except Exception:
        pr_auc = None
    return best_f1, best_th, pr_auc


def split_songs(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    # Heuristic: new song when bar value drops (reset)
    songs: list[list[dict[str, str]]] = []
    cur: list[dict[str, str]] = []
    prev_bar: int | None = None
    for r in rows:
        try:
            bar = int(r.get("bar", 0))
        except Exception:
            bar = 0
        if prev_bar is None or bar >= prev_bar:
            cur.append(r)
        else:
            if cur:
                songs.append(cur)
            cur = [r]
        prev_bar = bar
    if cur:
        songs.append(cur)
    return songs


def apply_hysteresis(probs: list[float], th_on: float, th_off: float) -> list[int]:
    preds: list[int] = []
    state = 0
    for p in probs:
        if state == 0 and p > th_on:
            state = 1
        elif state == 1 and p < th_off:
            state = 0
        preds.append(state)
    return preds


def enforce_min_gap(preds: list[int], probs: list[float], min_gap: int) -> list[int]:
    if min_gap <= 1:
        return preds
    last_pos = -min_gap - 1
    out = preds[:]
    for i, v in enumerate(preds):
        if v == 1:
            if i - last_pos < min_gap:
                # conflict: keep the higher-prob one
                if probs[i] > probs[last_pos] if 0 <= last_pos < len(probs) else True:
                    out[last_pos] = 0
                    last_pos = i
                else:
                    out[i] = 0
            else:
                last_pos = i
    return out


def viterbi_binary(probs: list[float], switch_cost: float = 1.0) -> list[int]:
    import math
    n = len(probs)
    if n == 0:
        return []
    log = lambda x: -1e9 if x <= 0 else math.log(x)
    e0 = [log(1 - p) for p in probs]
    e1 = [log(p) for p in probs]
    # transitions
    stay = 0.0
    switch = -abs(float(switch_cost))
    dp0 = [e0[0]]
    dp1 = [e1[0]]
    bt0 = [-1]
    bt1 = [-1]
    for t in range(1, n):
        a0 = dp0[t - 1] + stay
        b0 = dp1[t - 1] + switch
        if a0 >= b0:
            dp0.append(a0 + e0[t])
            bt0.append(0)
        else:
            dp0.append(b0 + e0[t])
            bt0.append(1)
        a1 = dp1[t - 1] + stay
        b1 = dp0[t - 1] + switch
        if a1 >= b1:
            dp1.append(a1 + e1[t])
            bt1.append(1)
        else:
            dp1.append(b1 + e1[t])
            bt1.append(0)
    # backtrack
    out = [0] * n
    state = 1 if dp1[-1] >= dp0[-1] else 0
    for t in reversed(range(n)):
        out[t] = state
        if t > 0:
            if state == 0:
                state = bt0[t]
            else:
                state = bt1[t]
    return out


def _stem(val: str) -> str:
    try:
        p = Path(val)
        if p.suffix:
            return p.stem
        return p.name or str(val)
    except Exception:
        return str(val)


def assign_song_ids_csv(
    rows: list[dict[str, object]],
    *,
    song_id_col: str | None = None,
    derive_from: str | None = None,
) -> list[str]:
    # Preference order: explicit song_id column -> derive from given path column -> derive from known path-like cols -> bar-reset heuristic
    if rows and song_id_col and song_id_col in rows[0]:
        return [str(r.get(song_id_col, "")) or "" for r in rows]
    # derive from a specified path column
    if rows and derive_from and derive_from in rows[0]:
        return [_stem(str(r.get(derive_from, ""))) for r in rows]
    # Auto-detect common path columns
    if rows:
        candidates = ["song_id", "file", "filename", "path", "source_path"]
        present = [c for c in candidates if c in rows[0]]
        if present:
            col = present[0]
            return [_stem(str(r.get(col, ""))) for r in rows]
    # Fallback: bar reset heuristic
    ids: list[str] = []
    song_idx = 0
    prev_bar: int | None = None
    for r in rows:
        try:
            bar = int(r.get("bar", 0))
        except Exception:
            bar = 0
        if prev_bar is not None and bar < prev_bar:
            song_idx += 1
        ids.append(f"S{song_idx + 1:04d}")
        prev_bar = bar
    return ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    device = setup_device(args.device)
    model, meta = reconstruct_model(args.ckpt, device)
    max_len = int(meta.get("max_len", args.max_len))

    # Load inputs: CSV or MIDI
    rows: list[dict[str, object]] = []
    source = None
    if args.in_csv:
        rows = read_csv_rows(args.in_csv)
        source = str(args.in_csv)
    elif args.in_midi:
        midi_paths = list_midi_files(args.in_midi)
        if not midi_paths:
            raise SystemExit(f"No MIDI files found under {args.in_midi}")
        pr = tuple(args.pitch_range) if args.pitch_range else None
        rows = midi_to_rows_batch(
            midi_paths,
            boundary_gap_beats=args.boundary_gap_beats,
            instrument=args.instrument,
            instrument_regex=args.instrument_regex,
            pitch_range=pr,
            path_hint=not bool(args.no_path_hint),
        )
        source = str(args.in_midi)
    else:
        raise SystemExit("Provide either --in-csv or --in-midi")

    # Threshold calibration if requested
    th_used = args.th if args.th is not None else 0.5
    calib = None
    if args.auto_th and args.val_csv and args.val_csv.is_file():
        val_rows = read_csv_rows(args.val_csv)
        val_groups = build_groups(val_rows)
        cols_val = set(val_rows[0].keys()) if val_rows else set()
        use_duv_val = "velocity_bucket" in cols_val and "duration_bucket" in cols_val
        use_bar_beat_val = bool(meta.get("use_bar_beat", False))
        use_harmony_val = bool(meta.get("use_harmony", False))
        probs_val, _ = predict(
            model,
            device,
            val_groups,
            max_len=max_len,
            use_duv_embed=use_duv_val,
            th=0.5,
            use_bar_beat=use_bar_beat_val,
            use_harmony=use_harmony_val,
        )
        trues_val = [int(r.get("boundary", 0)) for r in val_rows]
        best_f1, best_th, pr_auc = scan_thresholds(trues_val, probs_val, *tuple(args.scan_range))
        th_used = best_th
        calib = {"best_f1": best_f1, "best_th": best_th, "pr_auc": pr_auc}

    # Predict on input
    rows_str = [{k: str(v) for k, v in r.items()} for r in rows]
    groups = build_groups(rows_str)
    cols = set(groups[0][0].keys()) if groups and groups[0] else set()
    use_duv_embed = "velocity_bucket" in cols and "duration_bucket" in cols
    # Determine calibration temperature from checkpoint meta
    temp = None
    calib = meta.get("calibration") if isinstance(meta, dict) else None
    if isinstance(calib, dict) and "temperature" in calib:
        try:
            temp = float(calib["temperature"])
        except Exception:
            temp = None
    use_bar_beat = bool(meta.get("use_bar_beat", False))
    use_harmony = bool(meta.get("use_harmony", False))
    probs, _ = predict(
        model,
        device,
        groups,
        max_len=max_len,
        use_duv_embed=use_duv_embed,
        th=th_used,
        temp=temp,
        use_bar_beat=use_bar_beat,
        use_harmony=use_harmony,
    )

    # Flatten and write
    out_rows: list[dict[str, object]] = []
    i = 0
    # Prepare song_id
    if args.in_midi:
        song_ids = [str(r.get("song_id", "")) or _stem(str(r.get("path", ""))) for r in rows_str]
    else:
        # If --require-song-id, ensure we can derive from explicit columns
        if args.require_song_id and not (
            (args.song_id_col and args.song_id_col in rows_str[0])
            or (args.derive_song_id_from and args.derive_song_id_from in rows_str[0])
            or any(c in rows_str[0] for c in ["song_id", "file", "filename", "path", "source_path"])
        ):
            raise SystemExit(
                "--require-song-id is set, but no suitable column is present. Use --song-id-col or --derive-song-id-from."
            )
        song_ids = assign_song_ids_csv(rows_str, song_id_col=args.song_id_col, derive_from=args.derive_song_id_from)

    # Optional per-song adaptive threshold mapping
    per_song_th_map: dict[str, float] = {}
    if args.per_song_th:
        # case 1: input CSV has GT -> compute per-song thresholds directly
        if args.in_csv and rows_str and "boundary" in rows_str[0]:
            # Build sequences per song
            tmp: dict[str, list[int]] = {}
            probs_map: dict[str, list[float]] = {}
            for sid, r, p in zip(song_ids, rows_str, probs):
                tmp.setdefault(sid, []).append(int(r.get("boundary", 0)))
                probs_map.setdefault(sid, []).append(float(p))
            for sid in tmp:
                f1s, ths, _ = scan_thresholds(tmp[sid], probs_map[sid], *tuple(args.scan_range))
                per_song_th_map[sid] = ths
        # case 2: use mapping from val-csv if provided
        elif args.val_csv and args.val_csv.is_file():
            val_rows = read_csv_rows(args.val_csv)
            val_ids = assign_song_ids_csv(val_rows, song_id_col=args.song_id_col, derive_from=args.derive_song_id_from)
            val_groups = build_groups(val_rows)
            cols_val = set(val_rows[0].keys()) if val_rows else set()
            use_duv_val = "velocity_bucket" in cols_val and "duration_bucket" in cols_val
            use_bar_beat_val = bool(meta.get("use_bar_beat", False))
            use_harmony_val = bool(meta.get("use_harmony", False))
            probs_val, _ = predict(
                model,
                device,
                val_groups,
                max_len=max_len,
                use_duv_embed=use_duv_val,
                th=0.5,
                use_bar_beat=use_bar_beat_val,
                use_harmony=use_harmony_val,
            )
            # Compute per-song thresholds on val
            tmp: dict[str, list[int]] = {}
            probs_map: dict[str, list[float]] = {}
            for sid, r, p in zip(val_ids, val_rows, probs_val):
                tmp.setdefault(sid, []).append(int(r.get("boundary", 0)))
                probs_map.setdefault(sid, []).append(float(p))
            for sid in tmp:
                f1s, ths, _ = scan_thresholds(tmp[sid], probs_map[sid], *tuple(args.scan_range))
                per_song_th_map[sid] = ths

    # Optional post-processing per song (apply on probabilities before thresholding)
    # Build per-song probability vectors (in original row order)
    probs_by_song: dict[str, list[float]] = {}
    idx_by_song: dict[str, list[int]] = {}
    for idx, sid in enumerate(song_ids):
        probs_by_song.setdefault(sid, []).append(probs[idx])
        idx_by_song.setdefault(sid, []).append(idx)

    post_preds: dict[int, int] = {}
    th_on = float(args.th_on) if args.th_on is not None else th_used
    th_off = float(args.th_off) if args.th_off is not None else th_used
    # Hysteresis thresholds from args if provided via environment or defaults (optional future)
    # For now, we keep th_on=th_off unless per-song-th is used
    if args.per_song_th:
        # already have per_song_th_map
        pass
    # Apply CRF-like smoothing or hysteresis/min-gap per song
    for sid, pv in probs_by_song.items():
        indices = idx_by_song[sid]
        if len(pv) == 0:
            continue
        # Default path: hysteresis + min-gap
        preds_seq = apply_hysteresis(pv, th_on, th_off)
        preds_seq = enforce_min_gap(preds_seq, pv, min_gap=max(0, int(args.min_gap)))
        for j, idx in enumerate(indices):
            post_preds[idx] = int(preds_seq[j])

    for r, sid in zip(rows_str, song_ids):
        rr = dict(r)
        prob = float(probs[i]) if i < len(probs) else 0.0
        rr["pred_prob"] = prob
        th_row = per_song_th_map.get(sid, th_used)
        # start from post-processing decision; if per-song threshold differs, re-apply threshold
        pred_bin = post_preds.get(i, int(prob > th_used))
        if th_row != th_used:
            pred_bin = int(prob > th_row)
        rr["pred_boundary"] = int(pred_bin)
        rr["applied_th"] = float(th_row)
        rr.setdefault("song_id", sid)
        out_rows.append(rr)
        i += 1
    # Ensure stable field order (original + song_id + preds)
    base_fields = list(rows_str[0].keys()) if rows_str else []
    # place song_id after instrument if present
    if "song_id" not in base_fields:
        insert_after = "instrument" if "instrument" in base_fields else None
        if insert_after and insert_after in base_fields:
            idx = base_fields.index(insert_after) + 1
            base_fields = base_fields[:idx] + ["song_id"] + base_fields[idx:]
        else:
            base_fields.append("song_id")
    fieldnames = base_fields + ["pred_prob", "pred_boundary", "applied_th"]
    write_csv(out_rows, args.out_csv, fieldnames, append=bool(args.append))

    # Optional report: if ground-truth is present in input CSV, compute summary and by-song metrics
    report = {"source": source, "out_csv": str(args.out_csv), "th": th_used, "wrote": len(out_rows)}
    if calib:
        report["calibration"] = calib
    if args.in_csv and out_rows and "boundary" in out_rows[0]:
        trues = [int(r.get("boundary", 0)) for r in out_rows]
        best_f1, best_th, pr_auc = scan_thresholds(trues, [float(r["pred_prob"]) for r in out_rows], *tuple(args.scan_range))
        report["eval_on_input"] = {"best_f1": best_f1, "best_th": best_th, "pr_auc": pr_auc}
        # by-song using song_id grouping (preferred), fallback to bar-reset heuristic
        songs: list[list[dict[str, str]]] = []
        if "song_id" in out_rows[0]:
            groups_map: dict[str, list[dict[str, str]]] = {}
            for r in out_rows:
                sid = str(r.get("song_id", ""))
                groups_map.setdefault(sid, []).append({k: str(v) for k, v in r.items()})
            songs = [groups_map[k] for k in sorted(groups_map.keys())]
        else:
            songs = split_songs([{k: str(v) for k, v in r.items()} for r in out_rows])
        per_song = []
        f1_applied_map: dict[str, float] = {}
        for s in songs:
            y = [int(r.get("boundary", 0)) for r in s]
            p = [float(r.get("pred_prob", 0.0)) for r in s]
            f1s, ths, _ = scan_thresholds(y, p, *tuple(args.scan_range))
            sid = s[0].get("song_id", "") if s else ""
            # F1 at applied threshold (per-song or global)
            th_app = per_song_th_map.get(sid, th_used)
            preds_app = [1 if v > th_app else 0 for v in p]
            f1_app = f1_score(y, preds_app)
            f1_applied_map[sid] = f1_app
            per_song.append({"song_id": sid, "len": len(s), "best_f1": f1s, "best_th": ths, "applied_th": th_app, "applied_f1": f1_app})
        report["by_song"] = per_song

        # Visualizations (PR curve / confusion matrix) if requested
        if args.viz:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore
                from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay

                viz_dir = args.viz_dir or args.out_csv.parent
                viz_dir.mkdir(parents=True, exist_ok=True)
                probs_in = [float(r["pred_prob"]) for r in out_rows]
                preds_in = [1 if p > th_used else 0 for p in probs_in]
                prec, rec, _ = precision_recall_curve(trues, probs_in)
                plt.figure()
                plt.plot(rec, prec)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.tight_layout()
                pr_path = viz_dir / "pr_curve.png"
                plt.savefig(pr_path)
                plt.close()
                ConfusionMatrixDisplay.from_predictions(trues, preds_in)
                plt.tight_layout()
                cm_path = viz_dir / "confusion_matrix.png"
                plt.savefig(cm_path)
                plt.close()
                report.setdefault("viz_files", []).extend([str(pr_path), str(cm_path)])
            except Exception:
                pass

        # Timeseries visualization per song (limit N)
        if args.viz and args.viz_timeseries and args.viz_timeseries > 0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore
                viz_dir = args.viz_dir or args.out_csv.parent
                viz_dir.mkdir(parents=True, exist_ok=True)
                # Build song grouping
                songs_map: dict[str, list[dict[str, str]]] = {}
                for r in out_rows:
                    sid = str(r.get("song_id", ""))
                    songs_map.setdefault(sid, []).append({k: str(v) for k, v in r.items()})
                # Select up to N songs per ordering rule
                items = list(songs_map.items())
                if args.viz_sort == "len":
                    items.sort(key=lambda kv: -len(kv[1]))
                elif args.viz_sort == "f1_asc":
                    items.sort(key=lambda kv: f1_applied_map.get(kv[0], 0.0))
                else:  # f1_desc
                    items.sort(key=lambda kv: f1_applied_map.get(kv[0], 0.0), reverse=True)
                order = items[: args.viz_timeseries]
                for sid, rows_song in order:
                    probs_s = [float(r.get("pred_prob", 0.0)) for r in rows_song]
                    gts_s = [int(r.get("boundary", 0)) for r in rows_song]
                    xs = list(range(len(rows_song)))
                    plt.figure(figsize=(10, 2.5))
                    plt.plot(xs, probs_s, label="prob")
                    # mark GT boundaries
                    ys = probs_s
                    for x, y, b in zip(xs, ys, gts_s):
                        if b:
                            plt.axvline(x=x, color="r", alpha=0.2, linewidth=1)
                    plt.ylim(0.0, 1.0)
                    plt.title(f"song {sid} | len={len(rows_song)}")
                    plt.tight_layout()
                    ts_path = viz_dir / f"timeseries_{sid}.png"
                    plt.savefig(ts_path)
                    plt.close()
                    report.setdefault("viz_timeseries", []).append(str(ts_path))
            except Exception:
                pass

        # Multi-page PDF with per-song plots
        if args.viz and args.viz_pdf:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore
                from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
                from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay

                viz_dir = args.viz_dir or args.out_csv.parent
                viz_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = args.viz_pdf
                items = list(songs_map.items())
                if args.viz_sort == "len":
                    items.sort(key=lambda kv: -len(kv[1]))
                elif args.viz_sort == "f1_asc":
                    items.sort(key=lambda kv: f1_applied_map.get(kv[0], 0.0))
                else:
                    items.sort(key=lambda kv: f1_applied_map.get(kv[0], 0.0), reverse=True)
                with PdfPages(pdf_path) as pdf:
                    for sid, rows_song in items[: max(args.viz_timeseries, 0) or len(items)]:
                        y = [int(r.get("boundary", 0)) for r in rows_song]
                        p = [float(r.get("pred_prob", 0.0)) for r in rows_song]
                        th_app = per_song_th_map.get(sid, th_used)
                        preds_app = [1 if v > th_app else 0 for v in p]
                        # Figure layout: PR curve (top-left), CM (top-right), timeseries (bottom)
                        fig = plt.figure(figsize=(10, 6))
                        ax1 = plt.subplot2grid((2, 2), (0, 0))
                        ax2 = plt.subplot2grid((2, 2), (0, 1))
                        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                        try:
                            prec, rec, _ = precision_recall_curve(y, p)
                            ax1.plot(rec, prec)
                            ax1.set_xlabel("Recall")
                            ax1.set_ylabel("Precision")
                            ax1.set_title(f"PR | {sid}")
                        except Exception:
                            pass
                        try:
                            ConfusionMatrixDisplay.from_predictions(y, preds_app, ax=ax2)
                            ax2.set_title(f"CM | th={th_app:.2f}")
                        except Exception:
                            pass
                        xs = list(range(len(p)))
                        ax3.plot(xs, p, label="prob")
                        for x, b in enumerate(y):
                            if b:
                                ax3.axvline(x=x, color="r", alpha=0.2, linewidth=1)
                        ax3.set_ylim(0.0, 1.0)
                        ax3.set_title(f"Timeseries | len={len(p)}")
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                report.setdefault("viz_pdf", str(pdf_path))
            except Exception:
                pass

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
