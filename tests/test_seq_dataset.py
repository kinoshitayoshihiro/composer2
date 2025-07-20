from pathlib import Path

import pandas as pd

from data.articulation_dataset import SeqArticulationDataset, seq_collate


def test_seq_dataset(tmp_path: Path) -> None:
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
    ds = SeqArticulationDataset(csv)
    batch = seq_collate([ds[0], ds[1]])
    assert batch["pitch"].shape == (2, 2)
    assert batch["bucket"].shape == (2, 2)
    assert batch["pedal"].shape == (2, 2)
    assert batch["velocity"].shape == (2, 2)
    assert batch["qlen"].shape == (2, 2)
    assert batch["labels"].shape == (2, 2)
    import pytest

    torch = pytest.importorskip("torch")
    assert batch["pad_mask"].dtype is torch.bool
