import json
import os
import yaml
import logging

logger = logging.getLogger(__name__)

_STYLE_MAP: dict[str, dict[str, list[float]]] = {
    "soft": {"velocity": [40, 60, 80], "cc": [50, 80, 100]},
    "hard": {"velocity": [80, 100, 120], "cc": [90, 110, 127]},
}


def load_style_db(path: str | None = None) -> None:
    """Load style curves from a YAML or JSON file and merge into the DB."""
    if path is None:
        path = os.environ.get("STYLE_DB_PATH")
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.lower().endswith((".yaml", ".yml")):
                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load style DB %s: %s", path, exc)
        return
    if not isinstance(data, dict):
        return
    for name, curve in data.items():
        if not isinstance(curve, dict):
            continue
        entry: dict[str, list[float]] = {}
        vels = curve.get("velocity")
        ccs = curve.get("cc")
        if isinstance(vels, list) and 4 <= len(vels) <= 16:
            entry["velocity"] = [float(v) for v in vels]
        if isinstance(ccs, list) and 4 <= len(ccs) <= 16:
            entry["cc"] = [float(c) for c in ccs]
        _STYLE_MAP[name] = entry


def get_style_curve(name: str, default: dict | None = None) -> dict | None:
    """Return style curve or *default* if not defined."""
    return _STYLE_MAP.get(name, default)


# Load automatically from env path on import
load_style_db()
