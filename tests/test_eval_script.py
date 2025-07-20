from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("torch")
pytest.importorskip("jsonargparse")
pytest.skip("eval script depends on LightningCLI", allow_module_level=True)
