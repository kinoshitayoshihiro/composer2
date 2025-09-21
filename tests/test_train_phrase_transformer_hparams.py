from pathlib import Path

import pytest

pytest.importorskip("torch")

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

    captured: dict[str, float] = {}
    from torch import nn
    import inspect

    try:
        from models.phrase_transformer import PhraseTransformer as RealPT  # type: ignore
    except Exception:
        RealPT = tp._import_phrase_transformer()

    class SpyPT(nn.Module):  # type: ignore[misc]
        def __init__(self, d_model, max_len, *,
                     section_vocab_size=0, mood_vocab_size=0,
                     vel_bucket_size=0, dur_bucket_size=0,
                     nhead=8, num_layers=4, dropout=0.1, **kwargs):
            super().__init__()
            captured.update(nhead=nhead, num_layers=num_layers, dropout=dropout)

            try:
                sig = inspect.signature(RealPT.__init__)
                allowed = {
                    p.name
                    for p in sig.parameters.values()
                    if p.name != "self" and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                }
                fkwargs = {
                    k: v
                    for k, v in dict(
                        d_model=d_model,
                        max_len=max_len,
                        section_vocab_size=section_vocab_size,
                        mood_vocab_size=mood_vocab_size,
                        vel_bucket_size=vel_bucket_size,
                        dur_bucket_size=dur_bucket_size,
                        nhead=nhead,
                        num_layers=num_layers,
                        dropout=dropout,
                        **kwargs,
                    ).items()
                    if k in allowed
                }
            except Exception:
                fkwargs = dict(
                    d_model=d_model,
                    max_len=max_len,
                    section_vocab_size=section_vocab_size,
                    mood_vocab_size=mood_vocab_size,
                    vel_bucket_size=vel_bucket_size,
                    dur_bucket_size=dur_bucket_size,
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=dropout,
                    **kwargs,
                )

            self.inner = RealPT(**fkwargs)

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    monkeypatch.setattr(tp, "PhraseTransformer", SpyPT, raising=True)

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
        nhead=4,
        layers=2,
        dropout=0.2,
    )
    assert ckpt.is_file()
    assert captured == {"nhead": 4, "num_layers": 2, "dropout": 0.2}
