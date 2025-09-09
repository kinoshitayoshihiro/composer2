import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from scripts.segment_phrase import load_model, segment_bytes

CKPT = Path("checkpoints/epoch=0-step=2.ckpt")

@pytest.mark.parametrize("mid_path", sorted(Path("data/golden").glob("*.mid")))
def test_segment_regression(mid_path: Path) -> None:
    json_path = mid_path.with_suffix(".json")
    if not json_path.exists():
        pytest.skip(f"{json_path} missing")
    model = load_model("transformer", CKPT)
    res = segment_bytes(mid_path.read_bytes(), model, 0.5)
    expected = json.loads(json_path.read_text())
    assert res == expected

