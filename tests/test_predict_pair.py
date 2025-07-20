from __future__ import annotations

from pathlib import Path

import pytest
from music21 import note, stream

from utilities.ml_articulation import MLArticulationModel, predict

pytest.importorskip("torch")


def test_predict_pair() -> None:
    s = stream.Stream(
        [note.Note("C4", quarterLength=1.0), note.Note("D4", quarterLength=1.0)]
    )
    model = MLArticulationModel(num_labels=9)
    pairs = predict(s, model, Path("articulation_schema.yaml"), flat=False)
    assert len(pairs) == 2
    assert isinstance(pairs[0][0], note.Note)


def test_predict_flat() -> None:
    s = stream.Stream(
        [note.Note("E4", quarterLength=1.0), note.Note("F4", quarterLength=1.0)]
    )
    model = MLArticulationModel(num_labels=9)
    labels = predict(s, model)
    assert len(labels) == 2
