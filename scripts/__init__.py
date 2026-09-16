"""Repository command entry points.

Keep ``python -m scripts.<command>`` usable from a source checkout, including
Kaggle notebooks where editable-install ``.pth`` files may be disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
