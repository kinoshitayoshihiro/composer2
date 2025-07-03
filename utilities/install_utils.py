from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Sequence


def run_with_retry(cmd: Sequence[str], *, attempts: int = 3, delay: float = 5.0) -> None:
    """Run a shell command with retries."""
    for idx in range(attempts):
        try:
            subprocess.check_call(list(cmd))
            return
        except Exception as exc:  # pragma: no cover - network/install failures
            if idx == attempts - 1:
                raise
            logging.warning("command failed: %s; retrying", exc)
            time.sleep(delay)
