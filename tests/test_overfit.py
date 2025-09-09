from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from data.articulation_data import ArticulationDataModule
from ml_models.articulation_tagger import ArticulationTagger


def make_csv(path: Path) -> None:
    rows = []
    for i in range(10):
        rows.append(
            {
                "track_id": 0,
                "onset": i * 0.1,
                "pitch": 60,
                "duration": 1,
                "velocity": 80,
                "pedal_state": 0,
                "articulation_label": "legato" if i % 2 == 0 else "staccato",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_overfit(tmp_path: Path) -> None:
    csv = tmp_path / "artic.csv"
    make_csv(csv)
    schema = {"legato": 0, "staccato": 1}
    schema_path = tmp_path / "schema.yaml"
    yaml.safe_dump(schema, schema_path.open("w"))
    dm = ArticulationDataModule(csv, schema_path, batch_size=2, train_pct=1.0)
    dm.setup()
    model = ArticulationTagger(num_labels=len(schema))
    trainer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        batch = next(iter(dm.train_dataloader()))
        feats, labels, mask = batch
        trainer.zero_grad()
        emissions = model(*feats)
        loss = -model.crf(emissions, labels, mask)
        loss.backward()
        trainer.step()
    with torch.no_grad():
        batch = next(iter(dm.train_dataloader()))
        feats, labels, mask = batch
        emissions = model(*feats)
        pred = model.crf.decode(emissions, mask=mask)[0]
        y_true = labels[0][: len(pred)].tolist()
    assert f1_score(y_true, pred, average="macro") >= 0.95
