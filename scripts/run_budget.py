"""Day 6: run the attacker at real budgets and record the per-budget metrics.

Budgets are ATTACK-CANDIDATE budgets (generations). `queries_spent` is reported for
each, so every number can also be read per target query -- which is what the plan's
per-query metrics need and what makes sample efficiency legible. Each budget is an
independent run from the same fixed seed, so the table is reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_drift.config import get_config
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.metrics.coverage import (
    binning_sensitivity,
    maestro_coverage,
    owasp_coverage,
    unique_strategies,
)
from minimal_drift.runner.loop import AttackLoop


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="10,25,50,100")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--out", default=str(cfg.runs_dir / "budget_sweep"))
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for budget in budgets:
        env = AP2Env(agent=build_agent(args.profile))
        loop = AttackLoop(
            env=env,
            intents=list(env.world.intents),
            seeds=tuple(range(1, args.n_seeds + 1)),
            log_path=out / f"attempts_b{budget}.jsonl",
        )
        res = loop.run(budget, seed=cfg.seed)
        summary = res.archive["summary"]
        elites = res.archive["elites"]
        assert isinstance(summary, dict) and isinstance(elites, dict)

        filled = [tuple(k.split("|")) for k in elites]  # type: ignore[misc]
        op_kinds = {
            tuple(k.split("|")): [o["kind"] for o in v["delta"]["ops"]] for k, v in elites.items()
        }
        impacts = {tuple(k.split("|")): float(v["E"]) for k, v in elites.items()}
        q = max(res.queries_spent, 1)
        total_impact = sum(impacts.values())
        total_ate = sum(float(v["ate"]["ate"]) for v in elites.values())

        rows.append(
            {
                "budget_generations": budget,
                "generations_run": res.generations,
                "queries_spent": res.queries_spent,
                "wall_seconds": res.wall_seconds,
                "cells_filled": summary["filled"],
                "cells_filled_reachable": summary["filled_reachable"],
                "capacity_nominal": summary["capacity"],
                "coverage_nominal": summary["coverage"],
                "capacity_reachable": summary["reachable_capacity"],
                "coverage_reachable": summary["reachable_coverage"],
                "qd_score": summary["qd_score"],
                "median_delta_star": summary["median_delta_star"],
                "min_delta_star": summary["min_delta_star"],
                "unique_strategies": unique_strategies(filled),  # type: ignore[arg-type]
                "mechanisms_covered": summary["mechanisms_covered"],
                "successes": res.successes,
                "invalid_rejected": res.invalid_rejected,
                "non_attributable": res.non_attributable,
                "success_per_query": round(res.successes / q, 6),
                "ate_per_query": round(total_ate / q, 6),
                "impact_inr_per_query": round(total_impact / q, 4),
                "total_impact_inr": round(total_impact, 2),
                "time_to_first_success_s": res.time_to_first_success,
                "queries_to_first_success": res.queries_to_first_success,
                "reflections": res.reflections,
                "llm_calls": res.llm_calls,
                "owasp": owasp_coverage(filled).model_dump(mode="json"),  # type: ignore[arg-type]
                "owasp_coverage_in_scope": owasp_coverage(filled).coverage_of_in_scope,  # type: ignore[arg-type]
                "maestro": maestro_coverage(filled).model_dump(mode="json"),  # type: ignore[arg-type]
                "maestro_coverage_in_scope": maestro_coverage(filled).coverage_of_in_scope,  # type: ignore[arg-type]
                "binning": binning_sensitivity(
                    filled,  # type: ignore[arg-type]
                    op_kinds,  # type: ignore[arg-type]
                    impacts,  # type: ignore[arg-type]
                ).model_dump(mode="json"),
                "binning_spread": binning_sensitivity(filled, op_kinds, impacts).spread,  # type: ignore[arg-type]
                "notable": {k: v.model_dump(mode="json") for k, v in res.notable.items()},
                "archive": res.archive,
            }
        )
        print(
            f"budget {budget:4d}  gens={res.generations:4d} q={res.queries_spent:6d}  "
            f"cells={summary['filled']:2d}/{summary['capacity']} "
            f"(reach {summary['reachable_coverage']:.2f})  QD={summary['qd_score']:.3f}  "
            f"med|d*|={summary['median_delta_star']}  "
            f"succ/q={res.successes / q:.4f}  INR/q={total_impact / q:.2f}"
        )

    payload = {
        "model_id": rows[0] and build_agent(args.profile).model_id,
        "live_mode": cfg.live_mode,
        "seed": cfg.seed,
        "n_seeds": args.n_seeds,
        "tau": cfg.tau,
        "rho": cfg.rho,
        "rows": rows,
    }
    dest = out / "budget_sweep.json"
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"-> {dest}")
    return 0 if any(r["cells_filled"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
