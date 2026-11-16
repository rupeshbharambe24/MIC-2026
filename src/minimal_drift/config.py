"""Typed, environment-driven configuration: seeds, thresholds, and cost weights.

Every number the optimiser depends on lives here so a run is reproducible from
`(git sha, Config)`. No secrets are stored; API keys are read from the process
environment on demand.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"

load_dotenv(REPO_ROOT / ".env")


class CostWeights(BaseModel):
    """Weights for C(delta): the attacker's perturbation cost."""

    field: float = 1.0
    per_100_tokens: float = 0.25
    turn: float = 0.5
    surface: float = 1.0


class Config(BaseModel):
    """Global run configuration."""

    seed: int = Field(default_factory=lambda: int(os.getenv("MD_SEED", "1337")))
    n_seeds: int = Field(default_factory=lambda: int(os.getenv("MD_N_SEEDS", "12")))
    tau: float = Field(default_factory=lambda: float(os.getenv("MD_TAU", "0.30")))
    rho: float = Field(default_factory=lambda: float(os.getenv("MD_RHO", "0.50")))
    bootstrap_resamples: int = 2000
    cost_weights: CostWeights = Field(default_factory=CostWeights)

    catalog_path: Path = DATA_DIR / "catalog.json"
    intents_path: Path = DATA_DIR / "intents.json"
    runs_dir: Path = RUNS_DIR

    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    @property
    def gemini_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or None

    @property
    def alt_model_key(self) -> str | None:
        return os.getenv("ALT_MODEL_API_KEY") or None

    @property
    def heldout_model_key(self) -> str | None:
        return os.getenv("HELDOUT_MODEL_API_KEY") or None

    @property
    def live_mode(self) -> bool:
        """True when a real target model is reachable; otherwise surrogate mode."""
        return self.gemini_key is not None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config singleton."""
    return Config()
