"""Day 2 gate: measure and persist the clean-run baseline error epsilon per intent."""

from __future__ import annotations

import argparse
import json
import sys

from minimal_drift.config import get_config
from minimal_drift.harness.agent import build_agent
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.runner.baseline import clean_baseline


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=cfg.n_seeds)
    ap.add_argument("--profile", default="auto")
    args = ap.parse_args()

    env = AP2Env(agent=build_agent(args.profile))
    seeds = list(range(1, args.n_seeds + 1))
    stats = {i: clean_baseline(env, i, seeds) for i in env.world.intents}

    payload = {
        "model_id": getattr(env.agent, "model_id", "unknown"),
        "live_mode": cfg.live_mode,
        "n_seeds": args.n_seeds,
        "tau": cfg.tau,
        "per_intent": {k: v.model_dump(mode="json") for k, v in stats.items()},
        "mean_epsilon": round(sum(s.epsilon for s in stats.values()) / len(stats), 6),
        "max_epsilon": round(max(s.epsilon for s in stats.values()), 6),
        "mean_valid_rate": round(sum(s.valid_rate for s in stats.values()) / len(stats), 6),
    }
    out = cfg.runs_dir / "baseline_epsilon.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"model={payload['model_id']} n_seeds={args.n_seeds} tau={cfg.tau}")
    for k, s in stats.items():
        flag = "" if s.epsilon < cfg.tau else "  <-- EPSILON >= TAU"
        print(
            f"  {k:24s} eps={s.epsilon:.4f} ci95=[{s.ci95[0]:.4f},{s.ci95[1]:.4f}] "
            f"valid={s.valid_rate:.2f}{flag}"
        )
    print(f"mean eps={payload['mean_epsilon']:.4f} max eps={payload['max_epsilon']:.4f}")
    print(f"-> {out}")

    over = [k for k, s in stats.items() if s.epsilon >= cfg.tau]
    if over:
        print(f"FAIL: epsilon >= tau for {over}", file=sys.stderr)
        return 1
    if payload["mean_valid_rate"] < 1.0:
        print("FAIL: some clean runs did not produce a valid payment", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
