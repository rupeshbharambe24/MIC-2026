"""Shared fixtures: one world, one env, deterministic.

The suite is hermetic on purpose. `.env` may carry a real `GEMINI_API_KEY`, and
several modules build their agent with profile "auto", so without this the unit
tests would quietly start calling a metered, rate-limited API -- 12s per call on
the free tier, and flaky the moment the network is. Opt in with `MD_TEST_LIVE=1`
when you actually mean to exercise the live path.
"""

from __future__ import annotations

import os

import pytest

import minimal_drift.config  # noqa: F401 -- import first: it load_dotenv()s the real keys in

if os.getenv("MD_TEST_LIVE") != "1":
    for _key in ("GEMINI_API_KEY", "ALT_MODEL_API_KEY", "HELDOUT_MODEL_API_KEY"):
        os.environ.pop(_key, None)

from minimal_drift.catalog.ground_truth import default_world
from minimal_drift.harness.agent import SurrogateAgent
from minimal_drift.harness.ap2_env import AP2Env


@pytest.fixture(scope="session")
def world():
    return default_world()


@pytest.fixture()
def env(world):
    return AP2Env(world=world, agent=SurrogateAgent("surrogate-balanced"))
