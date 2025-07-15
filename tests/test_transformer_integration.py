from pathlib import Path

import pytest

try:
    import torch
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - skip if torch unavailable
    pytest.skip("torch required", allow_module_level=True)

from utilities.groove_transformer import GrooveTransformer, MultiPartDataset, collate_multi_part
from utilities import transformer_sampler


@pytest.mark.parametrize("parts", [["drums", "bass", "piano", "perc"]])
def test_train_and_sample(tmp_path: Path, parts: list[str]) -> None:
    sequences = []
    # create tiny synthetic dataset: repeated counting tokens
    for _ in range(4):
        seq = {}
        for p in parts:
            seq[p] = [1, 2, 3, 4] * 4
        sequences.append(seq)
    vocab_sizes = {p: 8 for p in parts}
    ds = MultiPartDataset(sequences, parts)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: {"input": collate_multi_part(b, parts), "target": collate_multi_part(b, parts)},
    )
    model = GrooveTransformer(vocab_sizes, d_model=32, nhead=2, num_layers=1)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_progress_bar=False)
    trainer.fit(model, dl)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    loaded = transformer_sampler.load(ckpt)
    events = transformer_sampler.sample_multi(loaded, {p: [0] for p in parts}, length=4, temperature={})
    for p in parts:
        assert events[p]
