"""Local LLM custom provider example.

Run from the repository root with the project venv::

    python examples\\llm_local\\workflow.py

The package is also importable (``import examples.llm_local.provider``) from
the repository root; the bootstrap below puts this directory, the repo root,
and ``src`` on ``sys.path`` so the flat sibling imports and ``memos.*``
resolve regardless of how the module is loaded. Scripts run directly
(``python examples\\llm_local\\workflow.py``) do not execute this file; they
rely on the script directory being on ``sys.path`` plus the editable
``memos`` install provided by the project venv.
"""

import os
import sys


_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLE_DIR, "..", ".."))
for _path in (_EXAMPLE_DIR, _REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
