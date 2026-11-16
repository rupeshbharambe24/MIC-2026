"""The attack loop.

Implements BUILD_PLAN §Day 5 verbatim, with the constraint order that makes the
target-query budget mean something:

    for generation in budget:
        cell    = selector.ucb_pick()
        delta   = mutator.mutate(archive.elite(cell) or seed, cell.operators)
        if not validate(apply(delta)).valid:          # HARD GATE, 0 target queries
            selector.update(cell, 0); archive.record_fail(cell, "invalid"); continue
        res     = ate(run_clean, run_pert, intent, n_seeds)
        if not res.attributable:
            selector.update(cell, 0); archive.record_fail(cell, boundary_reason); continue
        d_min   = shrink(delta, keep=lambda d: validate(d).valid and ate(...).attributable)
        R, S, E = transfer(d_min), stealth(d_min), economics(apply(d_min))
        if R >= rho:
            archive.upsert(cell, d_min, res, R, E)
        selector.update(cell, reward=res.attributable * (res.ate / cost(d_min)) * R)

Every attempt is written as one JSON line to `data/runs/`, carrying the fields §4
requires: cell, delta, valid, ATE, CI, |delta*|, R, S, E, boundary_reason, model_id, seed.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from minimal_drift.config import get_config
from minimal_drift.defenders import held_out, stealth, task_alignment
from minimal_drift.ensemble.transfer import Ensemble, build_ensemble, transfer
from minimal_drift.harness.ap2_env import AP2Env
from minimal_drift.mechanisms import MECHANISM_NAMES
from minimal_drift.mechanisms.base import Intervention
from minimal_drift.mechanisms.context import MechanismContext, build_context
from minimal_drift.minimizer.shrink import shrink_with_report
from minimal_drift.runner.evaluate import Evaluator
from minimal_drift.runner.jsonl import JsonlLogger
from minimal_drift.scoring.economics import impact
from minimal_drift.search.archive import Archive, Cell
from minimal_drift.search.mutator import REFLECTION, Mutator
from minimal_drift.search.selector import UCB1


class NotableAttack(BaseModel):
    """An attack worth reporting even if the archive's minimality rule evicted it."""

    cell_worked: tuple[str, str]
    descriptor: tuple[str, str]
    intent: str
    delta: Intervention
    delta_star: int
    cost: float
    ate: float
    ci95: tuple[float, float]
    R: float
    R_held_out: float
    S_in_loop: float
    S_held_out: float
    E: float
    promoted_to_elite: bool


class LoopResult(BaseModel):
    """Everything a run of the loop produced, plus what it cost."""

    budget: int
    generations: int
    queries_spent: int
    wall_seconds: float
    archive: dict[str, object]
    selector: dict[str, dict[str, float]]
    successes: int
    invalid_rejected: int
    non_attributable: int
    reflections: dict[str, int]
    notable: dict[str, NotableAttack]
    time_to_first_success: float | None
    queries_to_first_success: int | None
    llm_calls: int


@dataclass
class AttackLoop:
    """The orchestrator. Modules do not call each other sideways -- this wires them."""

    env: AP2Env
    intents: Sequence[str]
    seeds: Sequence[int] = field(default_factory=tuple)
    rho: float | None = None
    tau: float | None = None
    ensemble_arg: Ensemble | None = None
    use_llm: bool | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        cfg = get_config()
        self.tau = cfg.tau if self.tau is None else self.tau
        self.rho = cfg.rho if self.rho is None else self.rho
        self.seeds = tuple(self.seeds) or tuple(range(1, cfg.n_seeds + 1))
        self.intents = tuple(self.intents) or tuple(self.env.world.intents)
        self.archive = Archive(mechanisms=MECHANISM_NAMES)
        self.selector = UCB1(cells=self.archive.cells)
        self.mutator = Mutator(use_llm=self.use_llm)
        self.evaluator = Evaluator(env=self.env, seeds=self.seeds, tau=self.tau)
        self.ensemble: Ensemble = self.ensemble_arg or build_ensemble(self.env.world)
        self._ctx: dict[str, MechanismContext] = {}

    # -- helpers ----------------------------------------------------------- #
    def context(self, intent_id: str) -> MechanismContext:
        if intent_id not in self._ctx:
            self._ctx[intent_id] = build_context(
                self.env.world,
                intent_id,
                self.env.offers(intent_id, self.seeds[0]),
                self.tau or 0.3,
            )
        return self._ctx[intent_id]

    def _stealth(self, delta: Intervention, intent_id: str) -> tuple[float, float]:
        """S on both checkers over the perturbed arm."""
        spec = self.env.world.intent(intent_id)
        faithful = [self.env.world.product(s).refund_period for s in spec.ground_truth_skus]
        typical = len(self.context(intent_id).offers)
        il, ho = [], []
        for seed in self.seeds:
            r = self.env.run(intent_id, seed, stack=delta.stack(), keep_artifacts=True)
            assert r.view is not None
            label = (
                r.cart.contents.payment_request.details.display_items[0].label if r.cart else None
            )
            il.append(task_alignment.check(r.view, label))
            ho.append(
                held_out.check(
                    r.view,
                    r.chosen_cart_id,
                    (
                        self.env.world.product(r.purchased_sku).refund_period
                        if r.purchased_sku
                        else None
                    ),
                    faithful,
                    typical_depth=typical,
                )
            )
        return stealth(il), stealth(ho)

    def _economics(self, delta: Intervention, intent_id: str) -> float:
        vals = []
        for seed in self.seeds:
            r = self.env.run(intent_id, seed, stack=delta.stack(), keep_artifacts=True)
            if r.cart is not None and r.valid:
                vals.append(impact(r.cart, intent_id, self.env.world).impact)
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _keep(self, intent_id: str):
        """The FULL constraint set the minimiser must preserve: valid AND attributable."""

        def keep(candidate: Intervention) -> bool:
            if not candidate.ops:
                return False
            ev = self.evaluator.evaluate(candidate, intent_id)
            return ev.attributable

        return keep

    EXPLORE_EVERY = 4

    def _target_for(self, cell: Cell, generation: int) -> tuple[str, Intervention | None]:
        """Pick this generation's intent, and whether to refine the cell's elite.

        Refining means mutating the elite against ITS OWN intent: an elite carries cart
        ids, and mutating it against another intent's option set yields a delta whose
        targets do not exist there. But refining forever pins a cell to whichever intent
        first filled it, so every `EXPLORE_EVERY` generations the cell starts fresh on
        the next intent in rotation.
        """
        elite = self.archive.elite(cell)
        rotate = self.intents[generation % len(self.intents)]
        if elite is None or generation % self.EXPLORE_EVERY == 0:
            return rotate, None
        parent = elite.intervention
        if parent.intent_id not in self.intents:
            return rotate, None
        return parent.intent_id, parent

    # -- the loop ---------------------------------------------------------- #
    def run(
        self, budget: int, seed: int | None = None, max_queries: int | None = None
    ) -> LoopResult:
        """Spend `budget` generations, or stop early at `max_queries` target queries.

        One generation == one candidate evaluated. `queries_spent` is always reported,
        so a generation budget can be read as a query cost after the fact.
        """
        cfg = get_config()
        rng = random.Random(seed if seed is not None else cfg.seed)
        log = JsonlLogger(self.log_path or (cfg.runs_dir / "attempts.jsonl"))
        started = time.monotonic()
        successes = invalid = non_attr = 0
        reflections: dict[str, int] = {}
        # The archive's elite rule puts MINIMALITY first, so a high-impact attack can be
        # evicted by a cheaper one with a smaller payoff. That is the specified rule, but
        # it means the archive alone would hide the worst case -- so track it explicitly.
        notable: dict[str, NotableAttack] = {}
        first_success_t: float | None = None
        first_success_q: int | None = None

        generation = -1
        for generation in range(budget):
            if max_queries is not None and self.evaluator.queries >= max_queries:
                break
            cell: Cell = self.selector.ucb_pick()
            intent_id, parent = self._target_for(cell, generation)
            ctx = self.context(intent_id)
            boundary = self.archive.last_failure(cell) or ""
            if REFLECTION.get(boundary):
                reflections[boundary] = reflections.get(boundary, 0) + 1

            delta = self.mutator.mutate(parent, cell, rng, ctx, boundary_reason=boundary)

            if not delta.ops:
                self.selector.update(cell, 0.0)
                self.archive.record_fail(cell, "no_operators")
                continue

            ev = self.evaluator.evaluate(delta, intent_id)

            # HARD GATE: an invalid candidate never reaches the counterfactual stage.
            if not ev.valid:
                invalid += 1
                self.selector.update(cell, 0.0)
                self.archive.record_fail(cell, ev.boundary_reason or "invalid_payment")
                self._log(log, generation, cell, intent_id, delta, ev, None, None, None, None)
                continue

            if not ev.attributable:
                non_attr += 1
                self.selector.update(cell, 0.0)
                # A valid run that drifted on some seeds but did not clear tau is not
                # a "success"; recording it as one would send the wrong repair signal.
                reason = ev.boundary_reason or "no_drift"
                self.archive.record_fail(cell, "sub_threshold" if reason == "success" else reason)
                self._log(log, generation, cell, intent_id, delta, ev, None, None, None, None)
                continue

            # minimise, then measure transfer / stealth / impact on the MINIMISED delta
            delta_min, report = shrink_with_report(delta, self._keep(intent_id))
            res = self.evaluator.evaluate(delta_min, intent_id)
            ate_res = res.ate or ev.ate
            assert ate_res is not None

            tr = transfer(
                delta_min,
                intent_id,
                self.ensemble,
                seeds=self.seeds[: min(len(self.seeds), 8)],
                tau=self.tau,
                fitness_only=True,
            )
            s_in, s_held = self._stealth(delta_min, intent_id)
            econ = self._economics(delta_min, intent_id)

            promoted = False
            if tr.R >= (self.rho or 0.0):
                promoted = self.archive.upsert(
                    self.archive.descriptor(delta_min),
                    delta_min,
                    ate_res,
                    R=tr.R,
                    E=econ,
                    stealth_in_loop=s_in,
                    stealth_held_out=s_held,
                    delta_star=delta_min.n_fields,
                    generation=generation,
                )
            successes += 1
            if first_success_t is None:
                first_success_t = round(time.monotonic() - started, 3)
                first_success_q = self.evaluator.queries

            record = NotableAttack(
                cell_worked=cell,
                descriptor=self.archive.descriptor(delta_min),
                intent=intent_id,
                delta=delta_min,
                delta_star=delta_min.n_fields,
                cost=delta_min.cost(),
                ate=ate_res.ate,
                ci95=ate_res.ci95,
                R=tr.R,
                R_held_out=tr.R_held_out,
                S_in_loop=s_in,
                S_held_out=s_held,
                E=econ,
                promoted_to_elite=promoted,
            )
            top_impact = notable.get("max_impact")
            if top_impact is None or record.E > top_impact.E:
                notable["max_impact"] = record
            top_ate = notable.get("max_ate")
            if top_ate is None or record.ate > top_ate.ate:
                notable["max_ate"] = record
            cheapest = notable.get("min_delta_star")
            if cheapest is None or (record.delta_star, record.cost) < (
                cheapest.delta_star,
                cheapest.cost,
            ):
                notable["min_delta_star"] = record

            reward = (ate_res.ate / max(delta_min.cost(), 1e-9)) * tr.R
            self.selector.update(cell, reward)
            self._log(
                log,
                generation,
                cell,
                intent_id,
                delta_min,
                res,
                tr,
                s_in,
                s_held,
                econ,
                promoted=promoted,
                shrink_report=report.model_dump(mode="json"),
            )

        log.close()
        return LoopResult(
            budget=budget,
            generations=generation + 1,
            queries_spent=self.evaluator.queries,
            wall_seconds=round(time.monotonic() - started, 3),
            archive=self.archive.to_json(),
            selector=self.selector.state(),
            successes=successes,
            invalid_rejected=invalid,
            non_attributable=non_attr,
            reflections=reflections,
            notable=dict(notable),
            time_to_first_success=first_success_t,
            queries_to_first_success=first_success_q,
            llm_calls=self.mutator.llm_calls,
        )

    # -- logging ----------------------------------------------------------- #
    def _log(
        self,
        log: JsonlLogger,
        generation: int,
        cell: Cell,
        intent_id: str,
        delta: Intervention,
        ev,
        tr,
        s_in,
        s_held,
        econ,
        promoted: bool = False,
        shrink_report: dict | None = None,
    ) -> None:
        log.write(
            {
                "generation": generation,
                "cell_worked": list(cell),  # the niche this generation was spending on
                "descriptor": [delta.effective_mechanism, delta.stage],  # where it landed
                "intent": intent_id,
                "delta": delta.model_dump(mode="json"),
                "delta_fields": delta.n_fields,
                "delta_star": delta.n_fields,
                "cost": delta.cost(),
                "valid": ev.valid,
                "invalid_reasons": ev.invalid_reasons,
                "ate": ev.ate.ate if ev.ate else None,
                "ci": list(ev.ate.ci95) if ev.ate else None,
                "epsilon": ev.ate.baseline_error if ev.ate else None,
                "attributable": ev.attributable,
                "R": tr.R if tr else None,
                "R_held_out": tr.R_held_out if tr else None,
                "S_in_loop": s_in,
                "S_held_out": s_held,
                "E": econ,
                "boundary_reason": ev.boundary_reason,
                "promoted_to_elite": promoted,
                "shrink": shrink_report,
                "model_id": self.env.agent.model_id,
                "seed": get_config().seed,
                "queries_spent": self.evaluator.queries,
            }
        )
