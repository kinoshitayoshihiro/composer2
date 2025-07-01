import os

from setuptools import Extension, setup

USE_CYTHON = os.environ.get("MCY_USE_CYTHON", "1") == "1"
source = "cyext/humanize" + (".pyx" if USE_CYTHON else ".c")
ext_modules = [Extension("cyext.humanize", [source])]

if USE_CYTHON:
    try:
        from Cython.Build import cythonize
        ext_modules = cythonize(ext_modules, language_level="3")
    except Exception:
        pass

setup(name="cyext", ext_modules=ext_modules)
