"""Minimal PEP 517 backend to support offline editable installs in tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile

NAME = "modular-composer"
DIST_NAME = NAME.replace("-", "_")
VERSION = "0.9.0"
ROOT = Path(__file__).resolve().parent

METADATA = f"""Metadata-Version: 2.1
Name: {NAME}
Version: {VERSION}
Summary: Modular Composer editable install stub
"""

WHEEL = """Wheel-Version: 1.0
Generator: composer2-build-backend
Root-Is-Purelib: true
Tag: py3-none-any
"""


def get_requires_for_build_wheel(config_settings=None):  # pragma: no cover - trivial
    return []


def get_requires_for_build_editable(config_settings=None):  # pragma: no cover - trivial
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    dist = Path(metadata_directory) / f"{DIST_NAME}-{VERSION}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(METADATA, encoding="utf-8")
    (dist / "WHEEL").write_text(WHEEL, encoding="utf-8")
    (dist / "RECORD").write_text("", encoding="utf-8")
    return dist.name


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def _build_wheel(target_dir: Path, *, editable: bool) -> str:
    filename = f"{DIST_NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = target_dir / filename
    direct_url = {"url": str(ROOT.resolve())}
    with zipfile.ZipFile(wheel_path, "w") as zf:
        dist_base = f"{DIST_NAME}-{VERSION}.dist-info"
        zf.writestr(f"{dist_base}/METADATA", METADATA)
        zf.writestr(f"{dist_base}/WHEEL", WHEEL)
        zf.writestr(f"{dist_base}/RECORD", "")
        if editable:
            zf.writestr(f"{dist_base}/direct_url.json", json.dumps(direct_url) + "\n")
        zf.writestr(f"{DIST_NAME}.pth", str(ROOT.resolve()) + os.linesep)
    return filename


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(Path(wheel_directory), editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(Path(wheel_directory), editable=True)
