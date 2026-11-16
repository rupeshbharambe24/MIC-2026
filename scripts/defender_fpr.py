"""Measure both defenders' FALSE-POSITIVE rate on clean runs.

A defender's fire rate is meaningless without this number: an attack that "evades"
a checker firing on 100% of honest traffic has evaded nothing. Written to
`data/runs/defender_fpr.json` and quoted alongside every stealth figure.
"""

from __future__ import annotations

import argparse
import json

from minimal_drift.config import get_config
from minimal_drift.defenders import held_out, stealth, task_alignment
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=cfg.n_seeds)
    ap.add_argument("--profile", default="auto")
    args = ap.parse_args()

    env = AP2Env(agent=build_agent(args.profile))
    per_intent: dict[str, dict[str, object]] = {}
    inloop_all, heldout_all = [], []
    depths = []

    typical = {
        i: len(env.offers(i, seed=1)) for i in env.world.intents
    }  # this query's usual result depth, as a deployed monitor would know it

    for intent_id in env.world.intents:
        spec = env.world.intent(intent_id)
        faithful = [env.world.product(s).refund_period for s in spec.ground_truth_skus]
        il, ho = [], []
        for seed in range(1, args.n_seeds + 1):
            r = env.run(intent_id, seed, keep_artifacts=True)
            assert r.view is not None
            depths.append(len(r.view.options))
            label = (
                r.cart.contents.payment_request.details.display_items[0].label if r.cart else None
            )
            v1 = task_alignment.check(r.view, label)
            v2 = held_out.check(
                r.view,
                r.chosen_cart_id,
                env.world.product(r.purchased_sku).refund_period if r.purchased_sku else None,
                faithful,
                typical_depth=typical[intent_id],
            )
            il.append(v1)
            ho.append(v2)
        inloop_all += il
        heldout_all += ho
        per_intent[intent_id] = {
            "in_loop_fpr": round(sum(v.fired for v in il) / len(il), 6),
            "held_out_fpr": round(sum(v.fired for v in ho) / len(ho), 6),
            "in_loop_reasons": sorted({x for v in il for x in v.reasons}),
            "held_out_reasons": sorted({x for v in ho for x in v.reasons}),
        }

    payload = {
        "model_id": getattr(env.agent, "model_id", "unknown"),
        "n_runs": len(inloop_all),
        "in_loop_fpr": round(sum(v.fired for v in inloop_all) / len(inloop_all), 6),
        "held_out_fpr": round(sum(v.fired for v in heldout_all) / len(heldout_all), 6),
        "in_loop_clean_stealth": stealth(inloop_all),
        "held_out_clean_stealth": stealth(heldout_all),
        "clean_result_depth": {"min": min(depths), "max": max(depths)},
        "typical_depth_per_intent": typical,
        "per_intent": per_intent,
    }
    out = cfg.runs_dir / "defender_fpr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"clean runs: {payload['n_runs']}  result depth {payload['clean_result_depth']}")
    print(f"in-loop  defender FPR: {payload['in_loop_fpr']:.4f}")
    print(f"held-out defender FPR: {payload['held_out_fpr']:.4f}")
    for k, v in per_intent.items():
        if v["in_loop_fpr"] or v["held_out_fpr"]:
            print(f"  {k:24s} in={v['in_loop_fpr']:.2f} held={v['held_out_fpr']:.2f}")
            for reason in list(v["in_loop_reasons"]) + list(v["held_out_reasons"]):  # type: ignore[arg-type]
                print(f"      {reason}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
