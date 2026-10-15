"""
The package version must agree everywhere it is written (2026-10-15).

The 0.3.0 upload on PyPI reports ``jaxpolylog.__version__ == '0.2.1'`` while its metadata says
0.3.0: ``setup.py`` had been bumped and ``jaxpolylog/__init__.py`` had not. The documentation reads
``__version__`` from ``__init__.py``, so these two are the only places the version is written.
"""

import pathlib
import re

import jaxpolylog

SETUP_PY = pathlib.Path(__file__).resolve().parents[1] / "setup.py"


def test_setup_py_and_dunder_version_agree():
    match = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", SETUP_PY.read_text(), re.M)
    assert match, "no version= in setup.py"
    assert jaxpolylog.__version__ == match.group(1)
