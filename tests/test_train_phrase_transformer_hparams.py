from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torch.manual_seed(0)

from scripts import train_phrase as tp


@pytest.fixture
def dummy_transformer(monkeypatch: pytest.MonkeyPatch):
    """Provide a minimal PhraseTransformer stub for deterministic testing."""

    from torch import nn

    def _install(captured: dict[str, float]):
        class DummyTransformer(nn.Module):
            def __init__(
                self,
                d_model: int,
                max_len: int,
                *,
                nhead: int = 8,
                num_layers: int = 4,
                dropout: float = 0.1,
                **_: object,
            ) -> None:
                super().__init__()
                captured.update(
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                # Provide at least one parameter so optimizer construction stays valid.
                self._probe = nn.Linear(d_model, 1)

            def forward(self, feats, mask):
                if mask is None:
                    raise AssertionError("mask is required for shape resolution")
                batch, seq_len = mask.shape
                device = mask.device
                zeros = torch.zeros((batch, seq_len), dtype=torch.float32, device=device)
                pitch = torch.zeros((batch, seq_len, 128), dtype=torch.float32, device=device)
                return {
                    "boundary": zeros,
                    "vel_reg": zeros,
                    "dur_reg": zeros,
                    "pitch_logits": pitch,
                }

        monkeypatch.setattr(tp, "PhraseTransformer", DummyTransformer, raising=True)
        return DummyTransformer

    return _install


@pytest.mark.parametrize(
    ("nhead", "num_layers", "dropout"),
    [
        (2, 1, 0.0),
        (4, 2, 0.2),
        (8, 3, 0.1),
    ],
)
def test_transformer_hparams(
    tmp_path: Path,
    dummy_transformer,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> None:
    # This smoke test tracks hparam propagation and the forward signature only.
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

    captured: dict[str, float] = {}
    stub_cls = dummy_transformer(captured)

    ckpt = tmp_path / "m.ckpt"
    tp.train_model(
        train_csv,
        valid_csv,
        epochs=1,
        arch="transformer",
        out=ckpt,
        batch_size=1,
        d_model=32,
        max_len=32,
        nhead=nhead,
        layers=num_layers,
        dropout=dropout,
    )
    assert ckpt.is_file()
    assert captured == {"nhead": nhead, "num_layers": num_layers, "dropout": dropout}

    model = stub_cls(d_model=32, max_len=32, nhead=nhead, num_layers=num_layers, dropout=dropout)
    mask = torch.ones((1, 2), dtype=torch.bool)
    out = model({}, mask)
    assert out["boundary"].shape == (1, 2)
