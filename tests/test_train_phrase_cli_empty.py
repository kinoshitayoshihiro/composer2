import subprocess
import sys
from pathlib import Path


def test_cli_empty(tmp_path: Path) -> None:
    header = "pitch,velocity,duration,pos,boundary,bar\n"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_csv.write_text(header)
    val_csv.write_text(header)
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "from tests.torch_stub import _stub_torch; _stub_torch();"
        "from types import ModuleType;"
        "import sys;"
        "pt=ModuleType('models.phrase_transformer');"
        "pt.PhraseTransformer=type('PhraseTransformer', (), {});"
        "sys.modules.setdefault('models', ModuleType('models'));"
        "sys.modules['models.phrase_transformer']=pt;"
        "import runpy;"
        f"sys.argv=['scripts/train_phrase.py','{train_csv}','{val_csv}','--epochs','1','--out','out.ckpt'];"
        "runpy.run_path('scripts/train_phrase.py', run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode != 0
    assert "produced no usable rows" in result.stderr
