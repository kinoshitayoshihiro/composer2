from pathlib import Path

import pandas as pd
import pytest
import pytorch_lightning as pl

from data.articulation_datamodule import ArticulationDataModule
from ml_models.tagger_module import TaggerModule

pytestmark = pytest.mark.slow


def test_lightning_smoke(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 1],
            "onset": [0.0, 1.0, 0.0],
            "duration": [1.0, 1.0, 1.0],
            "velocity": [0.5, 0.5, 0.5],
            "pitch": [60, 62, 64],
            "pedal_state": [0, 0, 0],
            "bucket": [2, 2, 2],
            "articulation_label": [0, 1, 0],
        }
    )
    csv = tmp_path / "notes.csv"
    df.to_csv(csv, index=False)

    dm = ArticulationDataModule(csv_path=csv, batch_size=2, val_pct=0.5)
    model = TaggerModule(num_labels=9)
    trainer = pl.Trainer(fast_dev_run=True, enable_model_summary=False)
    trainer.fit(model, dm)
