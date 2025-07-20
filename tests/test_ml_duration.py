from pathlib import Path
import os
import pytest

if os.getenv("LIGHT") == "1":
    pytest.skip("Skip heavy tests in LIGHT mode", allow_module_level=True)


def test_overfit_duration(tmp_path):
    torch = pytest.importorskip('torch')
    pl = pytest.importorskip('pytorch_lightning')
    from types import SimpleNamespace
    from utilities.duration_datamodule import DurationDataModule
    from utilities.ml_duration import DurationTransformer, predict

    csv = Path('tests/data/duration_dummy.csv')
    cfg = SimpleNamespace(data=SimpleNamespace(csv=str(csv)), batch_size=1, max_len=24)
    dm = DurationDataModule(cfg)
    model = DurationTransformer(d_model=32, max_len=24)
    trainer = pl.Trainer(max_epochs=30, logger=False, enable_checkpointing=False)
    trainer.fit(model, dm)
    trainer.validate(model, datamodule=dm)
    assert float(trainer.callback_metrics['val_loss']) < 0.4
    feats, target, mask = next(iter(dm.train_dataloader()))
    pred = predict(feats, mask, model)
    loss = torch.nn.functional.l1_loss(pred[mask], target[mask])
    assert loss.item() < 0.35
