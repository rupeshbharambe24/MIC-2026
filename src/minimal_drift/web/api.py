"""FastAPI backend for the demo.

Serves the archive heatmap, the replayable end-to-end attack, and the live |δ*|
counter. Every number is recomputed from the harness on request, or read from a file a
measurement stage actually wrote -- there are no canned figures in this file, so the
demo cannot drift away from the test suite.

    make demo    # uvicorn minimal_drift.web.api:app --port 8000
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from minimal_drift.config import get_config
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.mechanisms import MECHANISM_NAMES
from minimal_drift.mechanisms.base import STAGES
from minimal_drift.metrics.coverage import maestro_coverage, owasp_coverage
from minimal_drift.validator import crypto
from minimal_drift.web import replay as replay_mod

UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(
    title="MINIMAL-DRIFT",
    description="Decision-boundary-aware, counterfactual, payment-valid red team for AP2.",
    version="0.1.0",
)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# The demo recomputes a real attack per request, so "auto" would put a metered model
# behind a public-ish page: slow, rate-limited, and dead once the daily quota is gone.
# Same call as the Makefile makes -- live is opt-in, never the default.
DEMO_PROFILE = os.getenv("MD_DEMO_PROFILE", "surrogate-balanced")


@lru_cache(maxsize=1)
def _env() -> AP2Env:
    return AP2Env(agent=build_agent(DEMO_PROFILE))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = UI_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "UI not built: src/minimal_drift/web/ui/index.html is missing")
    return page.read_text()


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """Run configuration and the guardrails, so the demo states its own limits."""
    cfg = get_config()
    return {
        "tau": cfg.tau,
        "rho": cfg.rho,
        "n_seeds": cfg.n_seeds,
        "seed": cfg.seed,
        "live_mode": cfg.live_mode,
        "model_id": _env().agent.model_id,
        "sign_mode": crypto.sign_mode(),
        "mechanisms": list(MECHANISM_NAMES),
        "stages": list(STAGES),
        "guardrails": [
            "Closed synthetic environment: public AP2 sample schema + synthetic catalog.",
            "No live payment rails, no real merchants, no real money.",
            "The attacker searches only INSIDE the valid payment space; "
            "invalid candidates are discarded before any target query.",
            "No secrets in code: all API keys come from the environment.",
        ]
        + (
            []
            if cfg.live_mode
            else [
                "NO LIVE MODEL KEY IS SET: the target is the deterministic surrogate agent, "
                "and transfer figures are over surrogate weight profiles, not across vendors."
            ]
        ),
    }


@app.get("/api/intents")
def intents() -> list[dict[str, Any]]:
    """Intents, with how attackable each one is."""
    env = _env()
    out = []
    for iid, spec in env.world.intents.items():
        cands = replay_mod.candidates(env, iid)
        out.append(
            {
                "id": iid,
                "text": spec.natural_language_description,
                "budget_cap": spec.budget_cap,
                "currency": spec.currency,
                "requires_refundability": spec.requires_refundability,
                "merchants": spec.merchants,
                "target_category": spec.target_category,
                "flagship": spec.flagship,
                "n_drift_targets": len(cands),
                "targets": [
                    {
                        "sku": c.product.sku,
                        "name": c.product.name,
                        "category": c.product.category,
                        "price": c.product.price,
                        "merchant": c.product.merchant,
                    }
                    for c in cands
                ],
            }
        )
    return out


@app.get("/api/replay")
def api_replay(intent: str = "intent_shoes_10k", target_sku: str | None = None) -> dict[str, Any]:
    """Recompute one full attack and narrate every stage. This is the demo."""
    try:
        return replay_mod.replay(intent, target_sku=target_sku).model_dump(mode="json")
    except KeyError:
        raise HTTPException(404, f"unknown intent {intent!r}") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/heatmap")
def heatmap() -> dict[str, Any]:
    """The archive as a mechanism x stage grid, coloured by best fitness."""
    cfg = get_config()
    sweep = _load(cfg.runs_dir / "budget_sweep" / "budget_sweep.json")
    if not sweep or not sweep.get("rows"):
        raise HTTPException(
            404, "no budget sweep found: run `make budget` (scripts/run_budget.py) first"
        )
    row = max(sweep["rows"], key=lambda r: r["budget_generations"])
    elites = row["archive"]["elites"]
    summary = row["archive"]["summary"]
    filled = [tuple(k.split("|")) for k in elites]

    cells = []
    for m in MECHANISM_NAMES:
        for s in STAGES:
            e = elites.get(f"{m}|{s}")
            cells.append(
                {
                    "mechanism": m,
                    "stage": s,
                    "filled": e is not None,
                    "fitness": e["fitness"] if e else 0.0,
                    "delta_star": e["delta_star"] if e else None,
                    "cost": e["cost"] if e else None,
                    "ate": e["ate"]["ate"] if e else None,
                    "ci95": e["ate"]["ci95"] if e else None,
                    "R": e["R"] if e else None,
                    "E": e["E"] if e else None,
                    "stealth_in_loop": e["stealth_in_loop"] if e else None,
                    "stealth_held_out": e["stealth_held_out"] if e else None,
                    "intent": e["delta"]["intent_id"] if e else None,
                    "ops": [o["kind"] for o in e["delta"]["ops"]] if e else [],
                    "reachable": True,
                }
            )
    reachable = {tuple(c) for c in summary.get("off_grid_cells", [])}
    for c in cells:
        key = (c["mechanism"], c["stage"])
        c["reachable"] = key not in reachable

    return {
        "budget": row["budget_generations"],
        "queries_spent": row["queries_spent"],
        "summary": summary,
        "cells": cells,
        "notable": row.get("notable", {}),
        "binning": row.get("binning", {}),
        "binning_spread": row.get("binning_spread"),
        "owasp": owasp_coverage(filled).model_dump(mode="json"),  # type: ignore[arg-type]
        "maestro": maestro_coverage(filled).model_dump(mode="json"),  # type: ignore[arg-type]
    }


@app.get("/api/budgets")
def budgets() -> list[dict[str, Any]]:
    """The per-budget results table."""
    cfg = get_config()
    sweep = _load(cfg.runs_dir / "budget_sweep" / "budget_sweep.json")
    if not sweep:
        raise HTTPException(404, "no budget sweep found: run `make budget` first")
    keep = (
        "budget_generations",
        "queries_spent",
        "cells_filled",
        "capacity_nominal",
        "coverage_nominal",
        "cells_filled_reachable",
        "capacity_reachable",
        "coverage_reachable",
        "qd_score",
        "median_delta_star",
        "min_delta_star",
        "unique_strategies",
        "success_per_query",
        "ate_per_query",
        "impact_inr_per_query",
        "time_to_first_success_s",
        "queries_to_first_success",
    )
    return [{k: r[k] for k in keep} for r in sweep["rows"]]


@app.get("/api/transfer")
def transfer() -> dict[str, Any]:
    cfg = get_config()
    data = _load(cfg.runs_dir / "transfer_eval.json")
    if not data:
        raise HTTPException(404, "no transfer eval found: run `make transfer` first")
    return data


@app.get("/api/baseline")
def baseline() -> dict[str, Any]:
    cfg = get_config()
    data = _load(cfg.runs_dir / "baseline_epsilon.json")
    if not data:
        raise HTTPException(404, "no baseline found: run `make baseline` first")
    return data


@app.get("/api/defenders")
def defenders() -> dict[str, Any]:
    cfg = get_config()
    data = _load(cfg.runs_dir / "defender_fpr.json")
    if not data:
        raise HTTPException(404, "no defender FPR found: run `make fpr` first")
    return data


@app.get("/api/blue")
def blue() -> dict[str, Any]:
    """Blue-gate detection vs the incumbent defenders, with both FPR denominators."""
    cfg = get_config()
    data = _load(cfg.runs_dir / "blue_eval.json")
    if not data:
        raise HTTPException(404, "no blue eval found: run `make blue` first")
    return data


@app.get("/api/inflation")
def inflation() -> dict[str, Any]:
    """Minimality inflation: |delta*| under the red predicate vs the blue predicate."""
    cfg = get_config()
    data = _load(cfg.runs_dir / "minimality_inflation.json")
    if not data:
        raise HTTPException(404, "no inflation run found: run `make inflation` first")
    return data


@app.get("/api/adaptive")
def adaptive() -> dict[str, Any]:
    """The adaptive search: what an attacker that knows the gate exists could find."""
    cfg = get_config()
    data = _load(cfg.runs_dir / "blue_adaptive.json")
    if not data:
        raise HTTPException(404, "no adaptive run found: run `make adaptive` first")
    return data


@app.get("/api/integrity")
def integrity() -> dict[str, Any]:
    """Confidence intervals, per-layer ablation, answer-key agreement, vacuous scope."""
    cfg = get_config()
    data = _load(cfg.runs_dir / "blue_integrity.json")
    if not data:
        raise HTTPException(404, "no integrity report found: run `make integrity` first")
    return data


@app.get("/api/probes")
def probes() -> dict[str, Any]:
    """Surface-8 protocol probes: identical-bundle replay and intent swap."""
    cfg = get_config()
    data = _load(cfg.runs_dir / "protocol_probes.json")
    if not data:
        raise HTTPException(404, "no protocol probes found: run `make probes` first")
    return data


@app.get("/api/report", response_class=PlainTextResponse)
def report() -> str:
    cfg = get_config()
    path = cfg.runs_dir / "report.md"
    if not path.exists():
        raise HTTPException(404, "no report found: run `make report` first")
    return path.read_text()
