from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, overload

import yaml

from utilities.settings import settings

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = object  # type: ignore

if torch is not None:
    try:
        from torchcrf import CRF  # type: ignore
    except ImportError:
        try:
            from torch_crf import CRF  # type: ignore
        except ImportError:  # pragma: no cover - final fallback
            from TorchCRF import CRF  # type: ignore
else:  # pragma: no cover - optional
    CRF = object  # type: ignore

import music21
import pretty_midi

from data.articulation_dataset import seq_collate
from ml_models import NoteFeatureEmbedder
from utilities.articulation_csv import extract_from_midi


class ArticulationTagger(nn.Module if torch is not None else object):
    """BiGRU-CRF model for note articulation tagging."""

    def __init__(
        self,
        num_labels: int,
        pitch_dim: int = 16,
        bucket_dim: int = 4,
        pedal_dim: int = 2,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch required")
        super().__init__()
        self.embed = NoteFeatureEmbedder(pitch_dim, bucket_dim, pedal_dim)
        d = pitch_dim + bucket_dim + pedal_dim + 2
        self.rnn = nn.GRU(d, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels)
        self._emissions: torch.Tensor | None = None

    def forward(
        self,
        pitch: torch.Tensor,
        bucket: torch.Tensor,
        pedal: torch.Tensor,
        velocity: torch.Tensor,
        qlen: torch.Tensor,
        labels: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(pitch, bucket, pedal, velocity, qlen)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        emissions = self.fc(x)
        self._emissions = emissions
        mask = pad_mask.bool() if pad_mask is not None else None
        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask)
            return -log_likelihood.mean()
        return emissions

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            batch["pitch"],
            batch["bucket"],
            batch["pedal"],
            batch["velocity"],
            batch["qlen"],
            labels=batch.get("labels"),
            pad_mask=batch.get("pad_mask"),
        )

    def decode(
        self,
        pitch: torch.Tensor | None = None,
        bucket: torch.Tensor | None = None,
        pedal: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        qlen: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if pitch is not None:
            emissions = self.forward(
                pitch, bucket, pedal, velocity, qlen, pad_mask=pad_mask
            )
        else:
            if self._emissions is None:
                raise RuntimeError("call forward() before decode")
            emissions = self._emissions
        mask = pad_mask.bool() if pad_mask is not None else None
        if hasattr(self.crf, "decode"):
            return self.crf.decode(emissions, mask)
        if hasattr(self.crf, "viterbi_decode"):
            return self.crf.viterbi_decode(emissions, mask)
        raise RuntimeError("CRF decode unavailable")

    def decode_batch(self, batch: dict[str, torch.Tensor]) -> list[list[int]]:
        _ = self.forward_batch(batch)
        return self.decode(pad_mask=batch.get("pad_mask"))


class MLArticulationModel(ArticulationTagger):
    @staticmethod
    def load(path: Path, schema: Path | None = None) -> MLArticulationModel:
        """Load a saved model with optional schema validation."""
        if torch is None:
            raise RuntimeError("torch required")

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}

        schema_labels: int | None = None
        if schema is not None:
            mapping = yaml.safe_load(Path(schema).read_text())
            schema_labels = len(mapping)

        state_labels = state.get("fc.weight").shape[0] if "fc.weight" in state else None

        if (
            state_labels is not None
            and schema_labels is not None
            and state_labels != schema_labels
        ):
            raise AssertionError(
                f"state-dict has {state_labels} labels but schema defines {schema_labels}"
            )

        num_labels = state_labels or schema_labels
        if num_labels is None:
            raise ValueError("could not determine number of labels")

        model = MLArticulationModel(num_labels)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


def load(
    path: Path,
    *,
    num_labels: int | None = None,
    schema: Path | None = None,
) -> MLArticulationModel:
    """Backward-compatible loader used in older tests."""
    if num_labels is not None:
        if torch is None:
            raise RuntimeError("torch required")
        model = MLArticulationModel(num_labels)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    return MLArticulationModel.load(path, schema)


@overload
def predict(
    score: music21.stream.Score,
    model: ArticulationTagger,
    schema_path: Path = settings.schema_path,
    *,
    flat: Literal[True] = True,
) -> list[str]: ...


@overload
def predict(
    score: music21.stream.Score,
    model: ArticulationTagger,
    schema_path: Path = settings.schema_path,
    *,
    flat: Literal[False],
) -> list[tuple[music21.note.Note, str]]: ...


def predict(
    score: music21.stream.Score,
    model: ArticulationTagger,
    schema_path: Path = settings.schema_path,
    *,
    flat: bool = True,
) -> list[str] | list[tuple[music21.note.Note, str]]:
    """Predict articulation labels for a score.

    When ``flat`` is ``True`` (default) return just the label list.  Otherwise
    return ``(note, label)`` pairs preserving the input note order.
    """

    try:
        pm = music21.midi.translate.m21ObjectToPrettyMIDI(score)
    except AttributeError:  # pragma: no cover - older music21
        mf = music21.midi.translate.streamToMidiFile(score)
        pm = pretty_midi.PrettyMIDI(io.BytesIO(mf.writestr()))
    df = extract_from_midi(pm)
    # Preserve dataclass rows so attribute access works during feature extraction
    rows = list(df.itertuples()) if hasattr(df, "itertuples") else list(df)
    batch = seq_collate([
        [
            {
                "pitch": int(r.pitch),
                "bucket": int(r.bucket),
                "pedal_state": int(r.pedal_state),
                "velocity": float(r.velocity),
                "qlen": float(r.duration),
                "label": 0,
            }
            for r in rows
        ]
    ])
    device = next(model.parameters()).device
    for k in batch:
        batch[k] = batch[k].to(device)

    ids = model.decode_batch(batch)[0]
    mapping = yaml.safe_load(Path(schema_path).read_text())
    inv = {v: k for k, v in mapping.items()}
    labels = [inv[i] for i in ids]
    if flat:
        return labels
    notes = list(score.flat.notes)
    return list(zip(notes, labels))


@overload
def predict_many(
    scores: list[music21.stream.Score],
    model: ArticulationTagger,
    *,
    flat: Literal[True] = True,
    schema_path: Path = settings.schema_path,
) -> list[list[str]]: ...


@overload
def predict_many(
    scores: list[music21.stream.Score],
    model: ArticulationTagger,
    *,
    flat: Literal[False],
    schema_path: Path = settings.schema_path,
) -> list[list[tuple[music21.note.Note, str]]]: ...


def predict_many(
    scores: list[music21.stream.Score],
    model: ArticulationTagger,
    *,
    flat: bool = True,
    schema_path: Path = settings.schema_path,
) -> list[list[str] | list[tuple[music21.note.Note, str]]]:
    """Predict articulations for multiple scores in a single batch."""

    pm_list = []
    note_lists: list[list[music21.note.Note]] = []
    for s in scores:
        try:
            pm = music21.midi.translate.m21ObjectToPrettyMIDI(s)
        except AttributeError:  # pragma: no cover - older music21
            mf = music21.midi.translate.streamToMidiFile(s)
            pm = pretty_midi.PrettyMIDI(io.BytesIO(mf.writestr()))
        df = extract_from_midi(pm)
        # Keep dataclass rows instead of converting to dicts
        rows = list(df.itertuples()) if hasattr(df, "itertuples") else list(df)
        pm_list.append(
            [
                {
                    "pitch": int(r.pitch),
                    "bucket": int(r.bucket),
                    "pedal_state": int(r.pedal_state),
                    "velocity": float(r.velocity),
                    "qlen": float(r.duration),
                    "label": 0,
                }
                for r in rows
            ]
        )
        note_lists.append(list(s.flat.notes))

    batch = seq_collate(pm_list)
    device = next(model.parameters()).device
    for k in batch:
        batch[k] = batch[k].to(device)

    ids_batch = model.decode_batch(batch)
    mapping = yaml.safe_load(Path(schema_path).read_text())
    inv = {v: k for k, v in mapping.items()}
    out = []
    for ids, notes in zip(ids_batch, note_lists):
        labels = [inv[i] for i in ids]
        if flat:
            out.append(labels)
        else:
            out.append(list(zip(notes, labels)))
    return out


__all__ = [
    "MLArticulationModel",
    "ArticulationTagger",
    "NoteFeatureEmbedder",
    "load",
    "predict",
    "predict_many",
]
