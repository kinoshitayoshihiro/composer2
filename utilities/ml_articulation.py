from __future__ import annotations

from pathlib import Path
from typing import List

from music21 import stream as m21stream
import torch

from data.articulation_data import ArticulationDataModule, pad_collate
from utilities.duration_bucket import to_bucket
from ml_models.articulation_tagger import ArticulationTagger


def load(model_path: str | Path, num_labels: int = 3) -> ArticulationTagger:
    model = ArticulationTagger(num_labels=num_labels)
    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def predict(score: m21stream.Score, model: ArticulationTagger) -> List[int]:
    notes = list(score.flat.notes)
    if not notes:
        return []
    pitch = torch.tensor([n.pitch.midi for n in notes], dtype=torch.long).unsqueeze(0)
    dur = torch.tensor([to_bucket(n.quarterLength) for n in notes], dtype=torch.long).unsqueeze(0)
    vel = torch.tensor([n.volume.velocity or 64 for n in notes], dtype=torch.float32).unsqueeze(0)
    pedal = torch.zeros_like(pitch, dtype=torch.long)
    emissions = model(pitch, dur, vel, pedal)
    mask = torch.ones(emissions.size()[:2], dtype=torch.bool)
    tags = model.crf.decode(emissions, mask=mask)[0]
    return tags

__all__ = ["load", "predict"]
