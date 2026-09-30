"""Model registry: scan weights/index dirs into a loadable model list.

Contract: ModelRegistry.scan() -> [{"pth", "index", "display"}, ...].
Pairing: one entry per *.pth; index pairs by same name, None when absent;
bb48k.pth pairs with guanguanV1.index (fixed, no same-name index).
Dir priority: scan() args > constructor args > env vars > in-project dirs
> RVC install dir. Env: RVC_ROOT / BSQ_RVC_ROOT, BSQ_WEIGHTS_DIR,
BSQ_INDEX_DIR.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

#: Project root (parent of the engine package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Portable fallback; machine-specific paths live in ignored runtime.local.txt.
DEFAULT_RVC_ROOT = PROJECT_ROOT / "RVC"

#: Fixed pairing: weights name (lowercase) -> index name (w/o .index suffix)
FIXED_INDEX_PAIRS = {"bb48k": "guanguanV1"}

ModelEntry = Dict[str, Optional[str]]


def rvc_root() -> Path:
    """RVC install root."""
    if (PROJECT_ROOT / "offline_bundle.json").is_file():
        return PROJECT_ROOT / "RVC"
    for name in ("BSQ_RVC_ROOT", "RVC_ROOT"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    runtime = os.environ.get("RVC_RUNTIME")
    if not runtime:
        local = PROJECT_ROOT / "runtime.local.txt"
        if local.is_file():
            runtime = local.read_text(encoding="utf-8-sig").strip()
    if runtime:
        return Path(runtime).parent.parent
    return Path(DEFAULT_RVC_ROOT)


def default_weights_dir() -> Path:
    """Weights dir: env var > in-project assets/weights > RVC install dir."""
    value = os.environ.get("BSQ_WEIGHTS_DIR")
    if value:
        return Path(value)
    local = PROJECT_ROOT / "assets" / "weights"
    if local.is_dir():
        return local
    return rvc_root() / "assets" / "weights"


def default_index_dir() -> Path:
    """Index dir: env var > in-project logs > RVC install dir."""
    value = os.environ.get("BSQ_INDEX_DIR")
    if value:
        return Path(value)
    local = PROJECT_ROOT / "logs"
    if local.is_dir():
        return local
    return rvc_root() / "logs"


class ModelRegistry:
    """Weights/index dir scanner; ModelRegistry.scan() also works unbound."""

    def __init__(self, weights_dir=None, index_dir=None):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.index_dir = Path(index_dir) if index_dir else None

    def scan(
        self=None,
        weights_dir=None,
        index_dir=None,
    ) -> List[ModelEntry]:
        """Scan models; missing args fall back to constructor/env/defaults."""
        if self is None:  # called as ModelRegistry.scan()
            self = ModelRegistry()
        wdir = (
            Path(weights_dir)
            if weights_dir
            else (self.weights_dir or default_weights_dir())
        )
        idir = (
            Path(index_dir)
            if index_dir
            else (self.index_dir or default_index_dir())
        )

        models: List[ModelEntry] = []
        for pth in sorted(wdir.glob("*.pth"), key=lambda p: p.name.lower()):
            stem = pth.stem
            index = idir / (FIXED_INDEX_PAIRS.get(stem.lower(), stem) + ".index")
            models.append(
                {
                    "pth": str(pth.resolve()),
                    "index": str(index.resolve()) if index.is_file() else None,
                    "display": stem,
                }
            )
        return models
