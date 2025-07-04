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


def export_mix_json(parts, path: str) -> None:
    """Export basic mixing metadata for ``parts`` to ``path``."""
    data = {}

    def _add(name: str, part) -> None:
        entry = {
            "extra_cc": getattr(part, "extra_cc", []),
        }
        meta = getattr(part, "metadata", None)
        if meta is not None:
            ir_file = getattr(meta, "ir_file", None)
            if ir_file:
                entry["ir_file"] = ir_file
        shaper = getattr(part, "tone_shaper", None)
        if shaper is not None and hasattr(shaper, "_selected"):
            entry["preset"] = shaper._selected
        data[name] = entry

    if isinstance(parts, dict):
        for k, v in parts.items():
            _add(k, v)
    else:
        _add(getattr(parts, "id", "part"), parts)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


load_mix_profiles()
