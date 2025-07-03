import logging
import os
import sys

from setuptools import Extension, setup

SOURCES = ["cyext/humanize", "cyext/generators"]
USE_CYTHON = os.environ.get("MCY_USE_CYTHON", "1") == "1"

if os.environ.get("MCY_USE_CYTHON") == "0":
    setup(name="cyext", ext_modules=[], package_data={})
    sys.exit(0)

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
    setup(
        name="cyext",
        ext_modules=ext_modules,
        package_data={"cyext": ["*.so"]} if ext_modules else {},
        extras_require={
            "test": ["scikit-learn>=1.3.0", "librosa>=0.10.0", "soundfile>=0.12.0"],
        },
    )
except Exception as exc:  # pragma: no cover - build may fail
    logging.warning("C extension build failed, using pure Python fallback: %s", exc)
    setup(
        name="cyext",
        ext_modules=[],
        package_data={},
        extras_require={
            "test": ["scikit-learn>=1.3.0", "librosa>=0.10.0", "soundfile>=0.12.0"],
        },
    )
