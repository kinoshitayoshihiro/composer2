import logging
import os

from setuptools import Extension, setup

SOURCES = ["cyext/humanize", "cyext/generators"]
USE_CYTHON = os.environ.get("MCY_USE_CYTHON", "1") == "1"

ext_modules = []
if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except Exception as exc:  # pragma: no cover - optional
        logging.warning("Cython unavailable, falling back to C sources: %s", exc)
    else:
        ext_modules = cythonize(
            [Extension(f"cyext.{s.split('/')[-1]}", [f"{s}.pyx"]) for s in SOURCES],
            language_level="3",
        )
if not ext_modules:
    ext_modules = [Extension(f"cyext.{s.split('/')[-1]}", [f"{s}.c"]) for s in SOURCES]

try:
    setup(name="cyext", ext_modules=ext_modules)
except Exception as exc:  # pragma: no cover - build may fail
    logging.warning("C extension build failed, using pure Python fallback: %s", exc)
    setup(name="cyext", ext_modules=[])
