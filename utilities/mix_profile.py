import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MIX_MAP: dict[str, dict] = {}


def load_mix_profiles(path: str | None = None) -> None:
    if path is None:
        path = os.environ.get("MIX_PROFILE_PATH")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fh:
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
            if ir_file is not None:
                entry["ir_file"] = str(Path(ir_file))
            rendered = getattr(meta, "rendered_wav", None)
            is_strings = any(
                kw in name.lower() for kw in ["violin", "cello", "viola", "bass", "strings"]
            )
            if rendered is not None and is_strings:
                entry["rendered_wav"] = str(rendered)
            fx_env = getattr(meta, "fx_envelope", None)
            if fx_env:
                entry["fx_envelope"] = fx_env
        shaper = getattr(part, "tone_shaper", None)
        if shaper is not None and hasattr(shaper, "_selected"):
            entry["preset"] = shaper._selected
            if not entry.get("ir_file"):
                try:
                    ir = shaper.get_ir_file(fallback_ok=True)
                except FileNotFoundError:
                    ir = None
                if ir is not None:
                    entry["ir_file"] = str(ir)
            if getattr(shaper, "fx_envelope", None):
                entry["fx_cc"] = shaper.fx_envelope
        data[name] = entry

    if isinstance(parts, dict):
        for k, v in parts.items():
            _add(k, v)
    else:
        _add(getattr(parts, "id", "part"), parts)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


load_mix_profiles()
