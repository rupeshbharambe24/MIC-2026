"""Day 5: run the attack loop end to end on a small budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_drift.config import get_config
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.runner.loop import AttackLoop


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--n-seeds", type=int, default=cfg.n_seeds)
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--out", default=str(cfg.runs_dir / "day5"))
    ap.add_argument("--intents", default="", help="comma-separated; default all")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = AP2Env(agent=build_agent(args.profile))
    intents = [i for i in args.intents.split(",") if i] or list(env.world.intents)

    loop = AttackLoop(
        env=env,
        intents=intents,
        seeds=tuple(range(1, args.n_seeds + 1)),
        log_path=out / "attempts.jsonl",
    )
    res = loop.run(args.budget, max_queries=args.max_queries)
    (out / "loop_result.json").write_text(json.dumps(res.model_dump(mode="json"), indent=2) + "\n")

    s = res.archive["summary"]
    print(f"model={env.agent.model_id} budget={args.budget} intents={len(intents)}")
    print(
        f"cells {s['filled']}/{s['capacity']} nominal (coverage {s['coverage']:.2f}); "
        f"{s['reachable_capacity']} reachable (coverage {s['reachable_coverage']:.2f})"
    )
    print(
        f"QD={s['qd_score']:.4f}  median |delta*|={s['median_delta_star']}  "
        f"min |delta*|={s['min_delta_star']}  off-grid cells={s['off_grid_cells']}"
    )
    print(f"mechanisms covered: {s['mechanisms_covered']}")
    print(f"stages covered:     {s['stages_covered']}")
    print(
        f"successes={res.successes} invalid_rejected={res.invalid_rejected} "
        f"non_attributable={res.non_attributable} queries={res.queries_spent} "
        f"wall={res.wall_seconds}s llm_calls={res.llm_calls}"
    )
    print(f"reflections triggered: {res.reflections}")
    print(
        f"time/queries to first success: {res.time_to_first_success}s / "
        f"{res.queries_to_first_success}"
    )
    print("elites:")
    for cell, e in res.archive["elites"].items():  # type: ignore[union-attr]
        print(
            f"  {cell:28s} |d*|={e['delta_star']} C={e['cost']:5.2f} ATE={e['ate']['ate']:.3f} "
            f"R={e['R']:.2f} S_in={e['stealth_in_loop']:.2f} S_held={e['stealth_held_out']:.2f} "
            f"E=INR {e['E']:,.0f} fit={e['fitness']:.3f}"
        )
    n = res.notable
    if n:
        print("notable (the archive's minimality rule can evict these):")
        for k in ("max_impact", "max_ate", "min_delta_star"):
            r = n.get(k)
            if r:
                print(
                    f"  {k:15s} {r.intent:20s} |d*|={r.delta_star} C={r.cost:5.2f} "
                    f"ATE={r.ate:.3f} R={r.R:.2f} R_held={r.R_held_out:.2f} "
                    f"S_in={r.S_in_loop:.2f} E=INR {r.E:,.0f} "
                    f"in_archive={r.promoted_to_elite}"
                )
                print(f"                  ops={[o.kind for o in r.delta.ops]}")
    print(f"-> {out}")
    return 0 if s["filled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
