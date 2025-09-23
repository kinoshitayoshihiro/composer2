from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts import train_phrase as tp


def test_transformer_hparams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "pitch": 60,
            "velocity": 64,
            "duration": 1,
            "pos": 0,
            "boundary": 1,
            "bar": 0,
            "instrument": "",
            "section": "",
            "mood": "",
        }
    ]
    train_csv = tmp_path / "train.csv"
    tp.write_csv(rows, train_csv)
    valid_csv = tmp_path / "valid.csv"
    tp.write_csv(rows, valid_csv)

    torch.manual_seed(0)
    torch.set_num_threads(1)

    captured: dict[str, float | int] = {}

    class DummyTransformer(torch.nn.Module):
        def __init__(
            self,
            *,
            d_model: int,
            max_len: int,
            nhead: int,
            num_layers: int,
            dropout: float,
            **_: object,
        ) -> None:
            super().__init__()
            captured.update(nhead=nhead, num_layers=num_layers, dropout=dropout)

        def forward(self, *args, **kwargs):
            return torch.zeros(1, 1)

    monkeypatch.setattr(tp, "PhraseTransformer", DummyTransformer, raising=True)

    ckpt = tmp_path / "m.ckpt"
    tp.train_model(
        train_csv,
        valid_csv,
        epochs=0,
        arch="transformer",
        out=ckpt,
        batch_size=1,
        d_model=32,
        max_len=32,
        nhead=4,
        layers=2,
        dropout=0.2,
    )
    assert captured == {"nhead": 4, "num_layers": 2, "dropout": 0.2}
