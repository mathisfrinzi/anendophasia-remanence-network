"""Les tests importent le code de src/ et scripts/ comme le font les scripts du dépôt."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("src", "scripts"):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def petit_cfg():
    """Configuration réduite commune (quelques milliers de paramètres) : les tests tournent en quelques secondes sur CPU."""
    import s3_tasks
    st = dict(s3_tasks.ST, d_model=32, n_layer=1, n_head=2, d_tok=16, spatial_ch=8)
    return s3_tasks.model_cfg(st)
