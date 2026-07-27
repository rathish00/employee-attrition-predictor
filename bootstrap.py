"""Adds src/ to sys.path so `import attrition_predictor` works without an
editable pip install. Import this first, before importing attrition_predictor,
from any script that lives outside src/ (notebooks/, app/, tests/).

Why this exists: `pip install -e .` needs network access to resolve build
dependencies in some sandboxed environments. This bootstrap makes the
package importable with zero install step. If you DO have network access,
`pip install -e .` is the more standard approach and this becomes a no-op
(src is already on the path via the install).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
