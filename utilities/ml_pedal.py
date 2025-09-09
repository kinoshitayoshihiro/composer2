from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import music21
import numpy as np
import pretty_midi

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from ml_models.pedal_model import PedalModel
from utilities.pedal_frames import HOP_LENGTH, SR, extract_from_midi


class MLPedalModel(PedalModel):
    @staticmethod
    def load(path: str) -> "MLPedalModel":
        if torch is None:
            raise RuntimeError("torch required")
        state = torch.load(path, map_location="cpu")
        model = MLPedalModel()
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def predict(self, ctx: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch required")
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(x).squeeze(0).sigmoid()
        return out.cpu().numpy()


def predict(
    score: music21.stream.Score, model: MLPedalModel
) -> list[tuple[float, int, int]]:
    """Return CC64 events (time, cc, value) predicted for *score*."""
    try:
        pm = music21.midi.translate.m21ObjectToPrettyMIDI(score)
    except AttributeError:  # pragma: no cover - old music21
        mf = music21.midi.translate.streamToMidiFile(score)
        pm = pretty_midi.PrettyMIDI(io.BytesIO(mf.writestr()))
    df = extract_from_midi(pm)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    ctx = df[chroma_cols + ["rel_release"]].values.astype("float32")
    prob = model.predict(ctx)
    step = HOP_LENGTH / SR
    events = []
    prev = 0
    for i, p in enumerate(prob):
        state = int(p >= 0.5)
        if state != prev:
            events.append((i * step, 64, 127 if state else 0))
            prev = state
    return events


__all__ = ["MLPedalModel", "predict"]
