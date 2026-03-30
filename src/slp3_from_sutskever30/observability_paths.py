from __future__ import annotations

import os
from pathlib import Path


def get_observability_dir(root: Path) -> Path:
    if os.environ.get("CIRCLECI"):
        return root / "observability" / "ci_latest"
    return root / "observability" / "local"
