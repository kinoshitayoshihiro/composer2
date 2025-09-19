from __future__ import annotations

"""Shared helpers for DUV (Duration/Velocity) inference."""

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch

REQUIRED_COLUMNS: set[str] = {"pitch", "velocity", "duration", "position"}
OPTIONAL_COLUMNS: set[str] = {
    "bar",
    "bar_phase",
    "beat_phase",
    "section",
    "mood",
    "vel_bucket",
    "dur_bucket",
    "track_id",
    "file",
    "start",
    "onset",
}

OPTIONAL_FLOAT32_COLUMNS: set[str] = {"bar_phase", "beat_phase", "start", "onset"}
OPTIONAL_INT32_COLUMNS: set[str] = {"bar", "track_id", "section", "mood", "vel_bucket", "dur_bucket"}
CSV_FLOAT32_COLUMNS: set[str] = {
    "velocity",
    "duration",
    "bar_phase",
    "beat_phase",
    "start",
    "onset",
    "q_onset",
    "q_duration",
}
CSV_INT32_COLUMNS: set[str] = {
    "pitch",
    "position",
    "bar",
    "track_id",
    "vel_bucket",
    "dur_bucket",
    "section",
    "mood",
}


def _missing_required(columns: Iterable[str]) -> list[str]:
    return sorted(REQUIRED_COLUMNS - set(columns))
def duv_verbose(flag: bool | None) -> bool:
    env = os.getenv("COMPOSER2_VERBOSE", "")
    if env.lower() not in {"", "0", "false", "no"}:
        return True
    return bool(flag)


def duv_sequence_predict(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    *,
    verbose: bool = False,
) -> dict[str, np.ndarray] | None:
    """Run phrase-level DUV inference when the checkpoint requires it."""

    if not getattr(model, "requires_duv_feats", False):
        return None

    df = df.reset_index(drop=True)

    missing = _missing_required(df.columns)
    if missing:
        raise RuntimeError(
            "DUV checkpoint requires phrase-level features for inference but the CSV is missing: "
            + ", ".join(missing)
        )

    if "bar" not in df.columns:
        import warnings

        warnings.warn("DUV requires bar segmentation for best results", RuntimeWarning)

    core: Any = getattr(model, "core", model)
    max_len = int(getattr(core, "max_len", getattr(model, "max_len", 16)))
    d_model = int(getattr(core, "d_model", getattr(model, "d_model", 0)))
    has_vel = bool(getattr(model, "has_vel_head", getattr(core, "head_vel_reg", None)))
    has_dur = bool(getattr(model, "has_dur_head", getattr(core, "head_dur_reg", None)))
    if not has_vel and not has_dur:
        return None

    n = len(df)
    vel_pred = np.zeros(n, dtype=np.float32)
    dur_pred = np.zeros(n, dtype=np.float32)
    vel_mask = np.zeros(n, dtype=bool)
    dur_mask = np.zeros(n, dtype=bool)

    group_cols = [c for c in ("track_id", "bar") if c in df.columns]
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = ((None, df),)

    missing_optional: set[str] = set()

    preview: dict[str, object] | None = None

    stderr = None
    if verbose:
        import sys

        stderr = sys.stderr

    optional_summary_printed = False

    for _, group in groups:
        if group.empty:
            continue
        g = group.sort_values("position")
        length = min(len(g), max_len)
        if length == 0:
            continue
        g = g.iloc[:length]
        idx = g.index.to_numpy(dtype=np.int64, copy=False)

        pitch_vals = torch.as_tensor(
            g["pitch"].to_numpy(dtype="int64", copy=False)[:length],
            dtype=torch.long,
            device=device,
        ).clamp_(min=0, max=127)
        pos_vals = torch.as_tensor(
            g["position"].to_numpy(dtype="int64", copy=False)[:length],
            dtype=torch.long,
            device=device,
        ).clamp_(max=max_len - 1)
        vel_vals = torch.as_tensor(
            g["velocity"].to_numpy(dtype="float32", copy=False)[:length],
            dtype=torch.float32,
            device=device,
        )
        dur_vals = torch.as_tensor(
            g["duration"].to_numpy(dtype="float32", copy=False)[:length],
            dtype=torch.float32,
            device=device,
        )

        pitch = torch.zeros(1, max_len, dtype=torch.long, device=device)
        pitch[:, :length] = pitch_vals
        pos = torch.zeros(1, max_len, dtype=torch.long, device=device)
        pos[:, :length] = pos_vals
        vel_in = torch.zeros(1, max_len, dtype=torch.float32, device=device)
        vel_in[:, :length] = vel_vals
        dur_in = torch.zeros(1, max_len, dtype=torch.float32, device=device)
        dur_in[:, :length] = dur_vals

        feat_dict: dict[str, torch.Tensor] = {
            "pitch": pitch,
            "pitch_class": pitch % 12,
            "position": pos,
            "velocity": vel_in,
            "duration": dur_in,
        }

        use_bar_beat = bool(getattr(core, "use_bar_beat", False))
        has_section = getattr(core, "section_emb", None) is not None
        has_mood = getattr(core, "mood_emb", None) is not None
        has_vel_bucket = getattr(core, "vel_bucket_emb", None) is not None
        has_dur_bucket = getattr(core, "dur_bucket_emb", None) is not None

        summary: dict[str, str] = {}

        if use_bar_beat:
            bar_phase = torch.zeros(1, max_len, dtype=torch.float32, device=device)
            beat_phase = torch.zeros(1, max_len, dtype=torch.float32, device=device)
            if "bar_phase" in g.columns:
                bar_phase[:, :length] = torch.as_tensor(
                    g["bar_phase"].to_numpy(dtype="float32", copy=False)[:length],
                    dtype=torch.float32,
                    device=device,
                )
                summary["bar_phase"] = "csv"
            else:
                missing_optional.add("bar_phase")
                summary["bar_phase"] = "zero_fill"
            if "beat_phase" in g.columns:
                beat_phase[:, :length] = torch.as_tensor(
                    g["beat_phase"].to_numpy(dtype="float32", copy=False)[:length],
                    dtype=torch.float32,
                    device=device,
                )
                summary["beat_phase"] = "csv"
            else:
                missing_optional.add("beat_phase")
                summary["beat_phase"] = "zero_fill"
            feat_dict["bar_phase"] = bar_phase
            feat_dict["beat_phase"] = beat_phase
        if has_section:
            section = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "section" in g.columns:
                section[:, :length] = torch.as_tensor(
                    g["section"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["section"] = "csv"
            else:
                missing_optional.add("section")
                summary["section"] = "zero_fill"
            feat_dict["section"] = section
        if has_mood:
            mood = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "mood" in g.columns:
                mood[:, :length] = torch.as_tensor(
                    g["mood"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["mood"] = "csv"
            else:
                missing_optional.add("mood")
                summary["mood"] = "zero_fill"
            feat_dict["mood"] = mood
        if has_vel_bucket:
            vel_bucket = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "vel_bucket" in g.columns:
                vel_bucket[:, :length] = torch.as_tensor(
                    g["vel_bucket"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["vel_bucket"] = "csv"
            else:
                missing_optional.add("vel_bucket")
                summary["vel_bucket"] = "zero_fill"
            feat_dict["vel_bucket"] = vel_bucket
        if has_dur_bucket:
            dur_bucket = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "dur_bucket" in g.columns:
                dur_bucket[:, :length] = torch.as_tensor(
                    g["dur_bucket"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["dur_bucket"] = "csv"
            else:
                missing_optional.add("dur_bucket")
                summary["dur_bucket"] = "zero_fill"
            feat_dict["dur_bucket"] = dur_bucket

        mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
        mask[:, :length] = True

        with torch.no_grad():
            outputs = model(feat_dict, mask=mask)

        if isinstance(outputs, dict):
            vel_out = outputs.get("velocity") or outputs.get("vel_reg")
            dur_out = outputs.get("duration") or outputs.get("dur_reg")
        elif isinstance(outputs, (tuple, list)):
            vel_out = outputs[0] if len(outputs) > 0 else None
            dur_out = outputs[1] if len(outputs) > 1 else None
        else:
            vel_out = outputs
            dur_out = None

        if verbose and preview is None:
            preview = {
                "seq_len": int(length),
                "d_model_effective": d_model or None,
                "features": sorted(feat_dict),
            }

        if verbose and not optional_summary_printed and summary and stderr is not None:
            print({"duv_optional_features": summary}, file=stderr)
            optional_summary_printed = True

        if vel_out is not None:
            v = torch.clamp(vel_out.squeeze(0), 0.0, 1.0).mul(127.0).round().clamp(1.0, 127.0)
            vel_pred[idx] = v.detach().cpu().numpy()[:length]
            vel_mask[idx] = True
        if dur_out is not None:
            d = torch.expm1(dur_out.squeeze(0)).clamp(min=0.0)
            dur_pred[idx] = d.detach().cpu().numpy()[:length]
            dur_mask[idx] = True

    if missing_optional and verbose:
        import warnings

        warnings.warn(
            "DUV checkpoint requested optional features missing from CSV: "
            + ", ".join(sorted(missing_optional))
            + "; using zero-filled tensors for inference",
            RuntimeWarning,
        )

    if verbose and preview is not None and stderr is not None:
        head = [float(v) for v in vel_pred[:8].tolist()]
        print(
            {
                "duv_preview": {
                    **preview,
                    "velocity_head": head,
                }
            },
            file=stderr,
        )

    return {
        "velocity": vel_pred,
        "velocity_mask": vel_mask,
        "duration": dur_pred,
        "duration_mask": dur_mask,
    }


__all__ = [
    "OPTIONAL_FLOAT32_COLUMNS",
    "OPTIONAL_INT32_COLUMNS",
    "OPTIONAL_COLUMNS",
    "CSV_FLOAT32_COLUMNS",
    "CSV_INT32_COLUMNS",
    "REQUIRED_COLUMNS",
    "duv_sequence_predict",
    "duv_verbose",
]
