"""Day 1 gate: one clean end-to-end AP2 flow to a signed, valid PaymentMandate."""

from __future__ import annotations

import argparse
import json
import sys

from minimal_drift.config import get_config
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.harness.surfaces import SurfaceStack, noop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", default="intent_shoes_10k")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", default="auto", help="auto|live|surrogate-*")
    args = ap.parse_args()

    cfg = get_config()
    env = AP2Env(agent=build_agent(args.profile))
    res = env.run(args.intent, args.seed, keep_artifacts=True)

    # Demonstrate all four injectable surfaces are wired, with a no-op stack.
    noop_stack = SurfaceStack()
    for s in ("metadata", "ranking", "tool", "memory"):
        noop_stack.add(s, noop)  # type: ignore[arg-type]
    noop_res = env.run(args.intent, args.seed, stack=noop_stack)

    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.runs_dir / "day1_clean_run.json"
    out.write_text(
        json.dumps(
            {
                "live_mode": cfg.live_mode,
                "model_id": res.model_id,
                "intent": args.intent,
                "seed": args.seed,
                "chosen_cart_id": res.chosen_cart_id,
                "purchased_sku": res.purchased_sku,
                "divergence": res.divergence,
                "valid": res.valid,
                "reasons": res.validation.reasons,
                "boundary_reason": res.boundary_reason,
                "cart": res.cart.model_dump(mode="json") if res.cart else None,
                "payment": res.payment.model_dump(mode="json") if res.payment else None,
                "noop_surfaces_identical": (
                    noop_res.chosen_cart_id == res.chosen_cart_id
                    and noop_res.divergence == res.divergence
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"model={res.model_id} live={cfg.live_mode}")
    print(f"chose {res.chosen_cart_id} ({res.purchased_sku}) D={res.divergence:.3f}")
    print(f"signed PaymentMandate valid={res.valid} reasons={res.validation.reasons}")
    print(
        f"no-op surface stack is behaviour-identical: "
        f"{noop_res.chosen_cart_id == res.chosen_cart_id}"
    )
    print(f"-> {out}")
    if not res.valid or res.divergence != 0.0:
        print("FAIL: clean run must produce a valid, non-divergent payment", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
