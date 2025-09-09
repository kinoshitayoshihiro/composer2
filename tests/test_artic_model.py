from pathlib import Path

import yaml
import pandas as pd
from music21 import stream, note
import torch

from utilities import ml_articulation
from data.articulation_data import ArticulationDataModule
from ml_models.articulation_tagger import ArticulationTagger

SCHEMA = {"legato": 0, "staccato": 1}


def _make_csv(path: Path) -> Path:
    rows = []
    for tid in range(10):
        for i in range(3):
            pitch = 60 + i
            vel = 80
            pedal = i % 2
            label = "legato" if pedal else "staccato"
            rows.append(
                {
                    "track_id": tid,
                    "onset": i * 0.5,
                    "pitch": pitch,
                    "duration": 1,
                    "velocity": vel,
                    "pedal_state": pedal,
                    "articulation_label": label,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_forward_shape(tmp_path: Path) -> None:
    csv_path = _make_csv(tmp_path / "a.csv")
    schema_path = tmp_path / "schema.yaml"
    yaml.safe_dump(SCHEMA, schema_path.open("w"))
    dm = ArticulationDataModule(csv_path, schema_path, batch_size=2, train_pct=1.0)
    dm.setup()
    model = ArticulationTagger(num_labels=len(SCHEMA))
    batch = next(iter(dm.train_dataloader()))
    emissions = model(*batch[0])
    assert emissions.shape[:2] == batch[0][0].shape


def test_predict_api(tmp_path: Path) -> None:
    model = ArticulationTagger(num_labels=len(SCHEMA))
    s = stream.Score()
    p = stream.Part()
    for i in range(3):
        n = note.Note(60 + i, quarterLength=1.0)
        n.volume.velocity = 80
        p.append(n)
    s.append(p)
    tags = ml_articulation.predict(s, model)
    assert len(tags) == 3
    assert hasattr(tags[0], "pitch")
    assert hasattr(tags[0], "dur")
