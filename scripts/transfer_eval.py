"""Day 6: zero-shot transfer of the strategy memory found on the primary target.

Takes the elites discovered by the budget sweep and replays each one UNCHANGED against
every ensemble member -- including the members fitness never saw and Environment B
(re-seeded prices, re-worded intents). No re-search, no re-tuning: that is what makes
it zero-shot.

Reported separately, because they answer different questions:
  R              over fitness-visible members -- the loop's own constraint
  R_held_out     over members the search never optimised against
  zero-shot rate the share of elites that survive on at least one held-out member
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_drift.config import get_config
from minimal_drift.ensemble.transfer import build_ensemble, transfer
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.mechanisms.base import Intervention


def load_elites(path: Path) -> list[tuple[str, Intervention, dict]]:
    """Read the elites out of the largest-budget row of a budget sweep."""
    data = json.loads(path.read_text())
    row = max(data["rows"], key=lambda r: r["budget_generations"])
    out = []
    for cell, e in row["archive"]["elites"].items():
        out.append((cell, Intervention.model_validate(e["delta"]), e))
    return out


def _rate(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(cfg.runs_dir / "budget_sweep" / "budget_sweep.json"))
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--profile", default="auto")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only the first N elites (0 = all). For metered targets: the "
        "live members cost n_seeds*2 calls per elite.",
    )
    ap.add_argument("--out", default=str(cfg.runs_dir / "transfer_eval.json"))
    args = ap.parse_args()

    sweep = Path(args.sweep)
    if not sweep.exists():
        print(f"FAIL: no budget sweep at {sweep}; run scripts/run_budget.py first")
        return 1

    env = AP2Env(agent=build_agent(args.profile))
    ensemble = build_ensemble(env.world)
    elites = load_elites(sweep)
    if not elites:
        print("FAIL: the sweep archive is empty")
        return 1

    n_available = len(elites)
    if args.limit:
        # A partial run must not be mistaken for the full archive downstream.
        print(f"NOTE: --limit {args.limit} of {len(elites)} elites; partial coverage.")
        elites = elites[: args.limit]
    seeds = tuple(range(1, args.n_seeds + 1))
    rows: list[dict[str, object]] = []
    print(f"ensemble members ({len(ensemble.members)}):")
    for m in ensemble.describe():
        tag = "held-out" if m["held_out"] else "fitness "
        print(f"  [{tag}] {m['name']:32s} kind={m['kind']}")
    if not cfg.live_mode:
        print(
            "NOTE: no live model keys are set. Model transfer below is over the "
            "surrogate weight-profile ensemble and is NOT a cross-vendor claim."
        )
    print()

    def summarise(done: list[dict[str, object]]) -> dict[str, object]:
        env_b = [m for r in done for m in r["members"] if m["kind"] == "environment"]  # type: ignore[union-attr]
        return {
            "model_id": env.agent.model_id,
            "live_mode": cfg.live_mode,
            # Set only when two genuinely different vendors answered, not merely when a
            # second key exists: ALT_MODEL_BASE_URL can point at the same vendor as GEMINI.
            "cross_vendor": ensemble.cross_vendor,
            "vendors": sorted(ensemble.vendors),
            "rho": cfg.rho,
            "n_seeds": len(seeds),
            "n_elites": len(done),
            "zero_shot_rate": _rate(sum(bool(r["zero_shot"]) for r in done), len(done)),
            "mean_R_held_out": _rate(sum(float(r["R_held_out"]) for r in done), len(done)),
            "n_surviving_rho_on_held_out": sum(
                1 for r in done if float(r["R_held_out"]) >= cfg.rho
            ),
            "environment_b_transfer_rate": _rate(
                sum(1 for m in env_b if m["attributable"]), len(env_b)
            ),
            "ensemble": ensemble.describe(),
            "rows": done,
            # A --limit run covers part of the archive, and so does a run that died part
            # way: report.md must not read either as the whole thing.
            "partial": bool(args.limit) or len(done) < n_available,
            "n_elites_available": n_available,
        }

    def write(payload: dict[str, object]) -> Path:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return dest

    for cell, delta, meta in elites:
        tr = transfer(delta, delta.intent_id, ensemble, seeds=seeds, tau=cfg.tau)
        rows.append(
            {
                "cell": cell,
                "intent": delta.intent_id,
                "delta_star": meta["delta_star"],
                "cost": meta["cost"],
                "home_ate": meta["ate"]["ate"],
                "R_fitness": tr.R,
                "R_held_out": tr.R_held_out,
                "R_all": tr.R_all,
                "zero_shot": tr.zero_shot,
                "members": [m.model_dump(mode="json") for m in tr.members],
            }
        )
        print(
            f"  {cell:28s} |d*|={meta['delta_star']} home ATE={meta['ate']['ate']:.3f}  "
            f"R_fit={tr.R:.2f} R_held={tr.R_held_out:.2f} R_all={tr.R_all:.2f} "
            f"zero_shot={tr.zero_shot}"
        )
        write(summarise(rows))  # a dropped connection must not cost the whole sweep

    payload = summarise(rows)
    dest = write(payload)

    print()
    print(f"elites evaluated:            {len(rows)}")
    print(f"zero-shot transfer rate:     {payload['zero_shot_rate']:.2f} (>=1 held-out member)")
    print(f"mean R over held-out:        {payload['mean_R_held_out']:.2f}")
    print(
        f"elites with R_held >= rho:   {payload['n_surviving_rho_on_held_out']}/{len(rows)} "
        f"(rho={cfg.rho})"
    )
    print(f"Environment B transfer rate: {payload['environment_b_transfer_rate']:.2f}")
    vendors = ", ".join(sorted(ensemble.vendors)) or "none"
    print(f"cross-vendor transfer:       {payload['cross_vendor']}  (vendors: {vendors})")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
