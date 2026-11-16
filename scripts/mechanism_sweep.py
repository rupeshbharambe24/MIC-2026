"""Day 4 gate: what can each mechanism do ON ITS OWN?

For every (mechanism, intent) pair, propose a few candidates from that mechanism
alone and record the best result: is the payment valid, is the drift attributable,
and does either defender fire. The output answers the Day-4 acceptance question --
"at least one per mechanism yields ATE >= tau for some intent" -- with evidence
rather than assertion, and shows where a mechanism is *structurally* insufficient
on its own.
"""

from __future__ import annotations

import argparse
import json
import random

from minimal_drift.config import get_config
from minimal_drift.defenders import held_out, stealth, task_alignment
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.mechanisms import MECHANISMS
from minimal_drift.mechanisms.context import build_context
from minimal_drift.runner.evaluate import Evaluator


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=cfg.n_seeds)
    ap.add_argument("--tries", type=int, default=6, help="proposals per (mechanism, intent)")
    ap.add_argument("--profile", default="auto")
    args = ap.parse_args()

    env = AP2Env(agent=build_agent(args.profile))
    ev = Evaluator(env=env, seeds=tuple(range(1, args.n_seeds + 1)))
    rows: list[dict[str, object]] = []

    for name, mech in MECHANISMS.items():
        for intent_id in env.world.intents:
            ctx = build_context(env.world, intent_id, env.offers(intent_id, 1), cfg.tau)
            rng = random.Random(f"{cfg.seed}|{name}|{intent_id}")
            best: dict[str, object] | None = None
            for t in range(args.tries):
                delta = mech.propose(None, rng, ctx)
                if not delta.ops:
                    continue
                res = ev.evaluate(delta, intent_id)
                il, ho = [], []
                for seed in ev.seeds:
                    r = env.run(intent_id, seed, stack=delta.stack(), keep_artifacts=True)
                    label = (
                        r.cart.contents.payment_request.details.display_items[0].label
                        if r.cart
                        else None
                    )
                    il.append(task_alignment.check(r.view, label))
                    ho.append(
                        held_out.check(
                            r.view,
                            r.chosen_cart_id,
                            (
                                env.world.product(r.purchased_sku).refund_period
                                if r.purchased_sku
                                else None
                            ),
                            [
                                env.world.product(s).refund_period
                                for s in env.world.intent(intent_id).ground_truth_skus
                            ],
                            typical_depth=len(ctx.offers),
                        )
                    )
                row = {
                    "mechanism": name,
                    "intent": intent_id,
                    "try": t,
                    "stage": delta.stage,
                    "delta_fields": delta.n_fields,
                    "cost": delta.cost(),
                    "ops": [o.kind for o in delta.ops],
                    "valid": res.valid,
                    "ate": res.ate.ate if res.ate else 0.0,
                    "lower_ci": res.ate.ci95[0] if res.ate else 0.0,
                    "epsilon": res.ate.baseline_error if res.ate else None,
                    "attributable": res.attributable,
                    "economics_inr": res.economics,
                    "boundary_reason": res.boundary_reason,
                    "stealth_in_loop": stealth(il),
                    "stealth_held_out": stealth(ho),
                    "note": delta.note,
                }
                key = (row["attributable"], row["ate"], -row["delta_fields"])
                if best is None or key > (best["attributable"], best["ate"], -best["delta_fields"]):
                    best = row
            if best:
                rows.append(best)

    out = cfg.runs_dir / "day4_mechanism_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model_id": getattr(env.agent, "model_id", "unknown"),
                "tau": cfg.tau,
                "n_seeds": args.n_seeds,
                "queries_spent": ev.queries,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"{'mechanism':12s} {'solo tau-reached':16s} best ATE   best intent")
    reached = {}
    for name in MECHANISMS:
        mine = [r for r in rows if r["mechanism"] == name]
        wins = [r for r in mine if r["attributable"]]
        top = max(mine, key=lambda r: r["ate"]) if mine else None
        reached[name] = bool(wins)
        best_intent = max(wins, key=lambda r: r["ate"])["intent"] if wins else "-"
        print(
            f"{name:12s} {'YES' if wins else 'no':16s} "
            f"{(top['ate'] if top else 0.0):8.4f}   {best_intent}"
        )
    print(f"\nper-(mechanism,intent) best rows: {len(rows)}   queries: {ev.queries}")
    print("attributable rows:")
    for r in sorted(
        (r for r in rows if r["attributable"]), key=lambda r: (-r["ate"], r["delta_fields"])
    ):
        print(
            f"  {r['mechanism']:12s} {r['intent']:24s} |d|={r['delta_fields']} "
            f"ATE={r['ate']:.3f} S_in={r['stealth_in_loop']:.2f} "
            f"S_held={r['stealth_held_out']:.2f} INR {r['economics_inr']:,.0f}"
        )
    print("\nnon-attributable mechanisms and why:")
    for name, ok in reached.items():
        if not ok:
            mine = [r for r in rows if r["mechanism"] == name]
            reasons = sorted({str(r["boundary_reason"]) for r in mine})
            print(f"  {name}: boundary reasons {reasons}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
