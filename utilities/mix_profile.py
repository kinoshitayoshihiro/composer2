import json
import os
import logging

logger = logging.getLogger(__name__)

_MIX_MAP: dict[str, dict] = {}


def load_mix_profiles(path: str | None = None) -> None:
    if path is None:
        path = os.environ.get("MIX_PROFILE_PATH")
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("Failed to load mix profiles %s: %s", path, exc)
        return
    if isinstance(data, dict):
        _MIX_MAP.update(data)


def get_mix_chain(name: str, default: dict | None = None) -> dict | None:
    return _MIX_MAP.get(name, default)


load_mix_profiles()
