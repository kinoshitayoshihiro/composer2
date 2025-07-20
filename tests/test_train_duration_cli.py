from pathlib import Path
import os
import subprocess
import pytest

if os.getenv("LIGHT") == "1":
    pytest.skip("Skip heavy tests in LIGHT mode", allow_module_level=True)


def test_train_duration_cli(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    out = tmp_path / 'model.ckpt'
    subprocess.check_call([
        'python', 'scripts/train_duration.py',
        '--data', 'tests/data/duration_dummy.csv',
        '--out', str(out),
        '--epochs', '1',
        '--max-len', '24',
        f'trainer.default_root_dir={tmp_path}'
    ])
    assert out.exists()
    ckpts = list(tmp_path.glob('**/epoch=*.ckpt'))
    assert ckpts
