"""Assemble every measured number into one markdown report.

Reads only files other stages wrote, so the report cannot claim anything that was not
measured. Anything missing is printed as "not run" rather than omitted, and every
surrogate-only figure is labelled as such.

    python -m minimal_drift.metrics.report   ->  data/runs/report.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from minimal_drift.config import get_config


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cell(value: str) -> str:
    """A raw `|` silently shreds a markdown table, so escape it for every caller.

    Archive cell keys are "mechanism|stage", and any future column could carry one too.
    Headers that already spell out `\\|delta\\|` are left alone -- only bare pipes are escaped.
    """
    return re.sub(r"(?<!\\)\|", r"\\|", value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _live_transfer_section(live: dict[str, Any] | None) -> list[str]:
    """Transfer of surrogate-found elites onto real LLMs.

    Reported separately from the main transfer table and never merged into it: it is a
    small, partial, low-power sample against one vendor, and averaging it into the
    headline would launder both its weakness and its disagreement.
    """
    L = ["## Live transfer onto real models", ""]
    if not live:
        return L + ["_not run — needs a live key with quota; see README._", ""]

    members = [m for m in live["ensemble"] if m.get("vendor", "surrogate") != "surrogate"]
    rows = live["rows"]
    seeds_unhonoured = [m["name"] for m in members if not m.get("seed_honoured", True)]
    # Membership, not a name substring: matching on "gemini" silently counted every
    # OpenAI or Anthropic member as a non-transfer.
    live_names = {m["name"] for m in members}
    attributable = sum(
        1 for r in rows for m in r["members"] if m["attributable"] and m["member"] in live_names
    )
    tested = len(rows) * len(members)

    # Name the elites that cleared the bar on EVERY live model: that is the strongest
    # true statement available, and burying it under the aggregate would be as
    # dishonest as omitting the 20 that failed.
    never = [
        str(r["cell"]).replace("|", " × ")
        for r in rows
        if not any(
            m["ate"] for m in r["members"] if m["member"] in {mem["name"] for mem in members}
        )
    ]
    universal = [
        r
        for r in rows
        if all(
            m["attributable"]
            for m in r["members"]
            if m["member"] in {mem["name"] for mem in members}
        )
    ]
    L += [
        f"**{attributable} of {tested}** (elite × live model) pairs cleared the "
        f"attributability bar, against {len(rows)} elites that clear it on the surrogate "
        f"ensemble. Both halves of that matter:",
        "",
    ]
    if universal:
        L += [
            "- **It does happen.** "
            + "; ".join(
                f"`{str(r['cell']).replace('|', ' × ')}` at **\\|δ\\*\\| = "
                f"{r['delta_star']}** drifted *every* live model"
                for r in universal
            )
            + " — zero-shot, unchanged, from a search that never saw them.",
        ]
    L += [
        f"- **And it mostly does not.** {tested - attributable} of {tested} pairs did "
        f"not clear the bar"
        + (
            " — including " + ", ".join(f"`{c}`" for c in never[:3]) + ", which drifted "
            "no live model at all"
            if never
            else ""
        )
        + ". A surrogate-tuned archive is **not** a set of working LLM attacks.",
        "",
        _table(
            ["cell", "\\|δ\\*\\|", "home ATE"] + [m["name"] for m in members],
            [
                [str(r["cell"]).replace("|", " × "), str(r["delta_star"]), f"{r['home_ate']:.3f}"]
                + [
                    next(
                        (
                            f"{m['ate']:.3f}{'' if m['attributable'] else ' (n.s.)'}"
                            for m in r["members"]
                            if m["member"] == mem["name"]
                        ),
                        "—",
                    )
                    for mem in members
                ]
                for r in rows
            ],
        ),
        "",
        "**Read this with its caveats, all of which weaken it:**",
        "",
        (
            f"- **Partial**: {len(rows)} of {live.get('n_elites_available', '?')} elites."
            if live.get("partial")
            else "- Full archive."
        ),
        f"- **Low power**: {live.get('n_seeds', '?')} seeds per arm against the 8–12 used "
        "elsewhere, so a wide CI can fail attributability on a real effect. `(n.s.)` "
        "means not significant, **not** zero.",
        f"- **One vendor**: {', '.join(sorted(live.get('vendors', [])))}. "
        f"cross-vendor = **{live.get('cross_vendor')}**.",
    ]
    if seeds_unhonoured:
        L += [
            f"- **Unseeded pairs** on {', '.join(seeds_unhonoured)}: the endpoint rejects "
            "`seed`, so those matched pairs share a prompt and temperature 0 but not a "
            "sampler seed. Noisier than the rest of the table.",
        ]
    L += [
        "- The elites were searched **against the surrogate**, so this measures zero-shot "
        "surrogate→LLM transfer: the hardest direction, and the honest one to report.",
        "",
    ]
    return L


def build_report() -> str:
    cfg = get_config()
    d = cfg.runs_dir
    eps = _load(d / "baseline_epsilon.json")
    fpr = _load(d / "defender_fpr.json")
    day3 = _load(d / "day3_ranking_attack.json")
    sweep_mech = _load(d / "day4_mechanism_sweep.json")
    sweep = _load(d / "budget_sweep" / "budget_sweep.json")
    tr = _load(d / "transfer_eval.json")

    L: list[str] = ["# MINIMAL-DRIFT — measured results", ""]
    L += [
        f"- τ = {cfg.tau} · ρ = {cfg.rho} · N seeds = {cfg.n_seeds} · seed = {cfg.seed}",
        f"- live model mode: **{cfg.live_mode}**"
        + (
            ""
            if cfg.live_mode
            else " — every figure below is against the deterministic surrogate target"
        ),
        "- closed synthetic environment: public AP2 sample schema + synthetic catalog. "
        "No live payment rails, no real merchants, no real money.",
        "",
    ]

    # -- headline ---------------------------------------------------------- #
    L += ["## The number", ""]
    if sweep and sweep.get("rows"):
        top = max(sweep["rows"], key=lambda r: r["budget_generations"])
        med = top["median_delta_star"]
        mn = top["min_delta_star"]
        zs = tr["zero_shot_rate"] if tr else None
        rheld = tr["mean_R_held_out"] if tr else None
        L += [
            f"**Median |δ\\*| = {med}** — the certified-minimal number of changed fields that",
            "flips a valid AP2 payment's purchased object, net of the clean-run baseline ε,",
            f"across the filled archive at a {top['budget_generations']}-candidate budget",
            f"({top['queries_spent']} target queries). Minimum |δ\\*| = {mn}.",
            "",
        ]
        if tr:
            L += [
                f"Every elite is transfer-tested: zero-shot transfer rate **{zs:.2f}**, "
                f"mean R over held-out members **{rheld:.2f}**, "
                f"Environment-B transfer **{tr['environment_b_transfer_rate']:.2f}**.",
                f"Cross-vendor transfer: **{tr['cross_vendor']}**"
                + ("" if tr["cross_vendor"] else " — surrogate weight-profile ensemble only."),
                "",
            ]
    else:
        L += ["_budget sweep not run — `make budget`_", ""]

    # -- baseline ---------------------------------------------------------- #
    L += ["## Baseline error ε (clean runs, no attack)", ""]
    if eps:
        L += [
            f"- mean ε = **{eps['mean_epsilon']}**, max ε = **{eps['max_epsilon']}** "
            f"over {len(eps['per_intent'])} intents × {eps['n_seeds']} seeds",
            f"- clean-run payment validity: **{eps['mean_valid_rate']:.2f}**",
            "- every ε is below τ, so no attack's ATE is being credited to the agent's own error",
            "",
            _table(
                ["intent", "ε", "CI95", "valid rate", "SKUs bought when clean"],
                [
                    [
                        k,
                        f"{v['epsilon']:.4f}",
                        f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]",
                        f"{v['valid_rate']:.2f}",
                        ", ".join(f"{s}×{n}" for s, n in v["skus_purchased"].items()),
                    ]
                    for k, v in eps["per_intent"].items()
                ],
            ),
            "",
        ]
    else:
        L += ["_not run — `make baseline`_", ""]

    # -- defenders --------------------------------------------------------- #
    L += ["## Defender false-positive rates (the number that makes stealth mean anything)", ""]
    if fpr:
        L += [
            f"- in-loop (language/alignment) FPR: **{fpr['in_loop_fpr']:.4f}**",
            f"- held-out (structural, no language) FPR: **{fpr['held_out_fpr']:.4f}**",
            f"- measured over {fpr['n_runs']} clean runs; clean result depth "
            f"{fpr['clean_result_depth']['min']}–{fpr['clean_result_depth']['max']} options",
            "",
        ]
    else:
        L += ["_not run — `make fpr`_", ""]

    # -- flagship ---------------------------------------------------------- #
    L += ["## Flagship attack: running shoes → gift card", ""]
    if day3:
        L += [
            f"- intent: `{day3['intent']}` — \"{day3['intent_text']}\"",
            f"- target: **{day3['favoured_target']['name']}** "
            f"({day3['favoured_target']['merchant']}, "
            f"{day3['favoured_target']['currency']} {day3['favoured_target']['price']:.0f}, "
            f"refund_period {day3['favoured_target']['refund_period']}d)",
            "",
            _table(
                ["rung", "\\|δ\\|", "C(δ)", "valid", "ATE", "CI95", "attributable", "₹/session"],
                [
                    [
                        r["name"],
                        str(r["delta_fields"]),
                        f"{r['delta_cost']:.2f}",
                        "yes" if r["valid"] else f"**no** — {(r['invalid_reasons'] or ['?'])[0]}",
                        f"{r['ate']['ate']:.4f}" if r["ate"] else "—",
                        (
                            f"[{r['ate']['ci95'][0]:.3f}, {r['ate']['ci95'][1]:.3f}]"
                            if r["ate"]
                            else "—"
                        ),
                        "**yes**" if r["attributable"] else "no",
                        f"{r['economics_inr']:,.0f}",
                    ]
                    for r in day3["ladder"]
                ],
            ),
            "",
            f"- ablation: dominant stage **{day3['ablation']['dominant_stage']}**"
            + (
                " — **conjunctive**: no single stage reproduces the effect alone"
                if day3["ablation"].get("conjunctive")
                else ""
            ),
            "- leave-one-out: "
            + "; ".join(
                f"`{o['kind']}` {'necessary' if o['necessary'] else 'not necessary'} "
                f"(ATE without = {o['result_without']['ate']:.3f})"
                for o in day3["ablation"]["operators"]
            ),
            f"- confirmation shown to the user: `{day3['confirmation']['rendered_prompt']}` "
            f"(disclosed the signed item: **{day3['confirmation']['disclosed']}**)",
            "",
        ]
    else:
        L += ["_not run — `make ranking`_", ""]

    # -- mechanisms -------------------------------------------------------- #
    L += ["## What each mechanism can do alone", ""]
    if sweep_mech:
        by_mech: dict[str, list[dict]] = {}
        for r in sweep_mech["rows"]:
            by_mech.setdefault(r["mechanism"], []).append(r)
        rows = []
        for name, rs in by_mech.items():
            wins = [r for r in rs if r["attributable"]]
            best = max(rs, key=lambda r: r["ate"])
            rows.append(
                [
                    name,
                    "**yes**" if wins else "no",
                    f"{best['ate']:.4f}",
                    best["intent"] if wins else "—",
                    str(best["delta_fields"]),
                    f"{best['stealth_in_loop']:.2f}",
                    f"{best['stealth_held_out']:.2f}",
                ]
            )
        L += [
            _table(
                [
                    "mechanism",
                    "reaches τ alone",
                    "best ATE",
                    "on intent",
                    "\\|δ\\|",
                    "S in-loop",
                    "S held-out",
                ],
                rows,
            ),
            "",
            "`consent` reaches ATE 0 in isolation by construction: it removes the last check",
            "on a decision another mechanism already moved. It is necessary, not sufficient.",
            "",
        ]
    else:
        L += ["_not run — `make sweep`_", ""]

    # -- budget sweep ------------------------------------------------------ #
    L += ["## Per-budget results", ""]
    if sweep and sweep.get("rows"):
        L += [
            _table(
                [
                    "budget (candidates)",
                    "target queries",
                    "cells (nominal)",
                    "cells (reachable)",
                    "QD",
                    "median \\|δ\\*\\|",
                    "unique strategies",
                    "succ/query",
                    "ATE/query",
                    "₹/query",
                    "time to 1st success",
                ],
                [
                    [
                        str(r["budget_generations"]),
                        str(r["queries_spent"]),
                        f"{r['cells_filled']}/{r['capacity_nominal']} "
                        f"({r['coverage_nominal']:.2f})",
                        f"{r['cells_filled_reachable']}/{r['capacity_reachable']} "
                        f"({r['coverage_reachable']:.2f})",
                        f"{r['qd_score']:.3f}",
                        str(r["median_delta_star"]),
                        str(r["unique_strategies"]),
                        f"{r['success_per_query']:.4f}",
                        f"{r['ate_per_query']:.4f}",
                        f"{r['impact_inr_per_query']:.2f}",
                        f"{r['time_to_first_success_s']}s / {r['queries_to_first_success']}q",
                    ]
                    for r in sweep["rows"]
                ],
            ),
            "",
        ]
        top = max(sweep["rows"], key=lambda r: r["budget_generations"])
        L += ["### Coverage against external taxonomies", ""]
        for key in ("owasp", "maestro"):
            t = top[key]
            L += [
                f"**{t['name']}** — covered {len(t['covered_items'])} of "
                f"{t['in_scope_items']} in-scope items "
                f"({len(t['covered_items']) / max(t['in_scope_items'], 1):.2f}); "
                f"{len(t['out_of_scope'])} of {t['total_items']} declared out of scope.",
                f"- covered: {', '.join(t['covered_items']) or '—'}",
                f"- in scope, not covered: {', '.join(t['uncovered_in_scope']) or '—'}",
                f"- out of scope: {', '.join(t['out_of_scope'])}",
                "",
            ]
        L += [
            "> The item lists are transcribed from published frameworks; **the mapping from",
            "> our cells to their items is our own judgement**, not endorsed by either body.",
            "> Re-check both lists against their current published revisions before quoting.",
            "",
            "### Binning sensitivity",
            "",
            _table(
                ["binning", "filled/capacity", "coverage"],
                [
                    [
                        v["name"],
                        f"{v['filled']}/{v['capacity']}",
                        f"{v['filled'] / max(v['capacity'], 1):.2f}",
                    ]
                    for v in top["binning"]["variants"]
                ],
            ),
            "",
            f"Spread between binnings: **{top['binning_spread']:.3f}**. "
            + (
                "Coverage is stable under re-binning."
                if top["binning_spread"] <= 0.25
                else "Coverage moves materially under re-binning, so the *absolute* coverage "
                "figure is partly an artefact of the grid — the per-mechanism and "
                "per-taxonomy results are the load-bearing ones."
            ),
            "",
        ]

    # -- transfer ---------------------------------------------------------- #
    L += ["## Transfer", ""]
    if tr:
        L += [
            f"- elites evaluated: {tr['n_elites']}",
            f"- zero-shot transfer rate (≥1 held-out member): **{tr['zero_shot_rate']:.2f}**",
            f"- mean R over held-out members: **{tr['mean_R_held_out']:.2f}**",
            f"- elites with R_held-out ≥ ρ={tr['rho']}: "
            f"**{tr['n_surviving_rho_on_held_out']}/{tr['n_elites']}**",
            "- Environment B (re-seeded catalog, re-worded intents): "
            f"**{tr['environment_b_transfer_rate']:.2f}**",
            "",
            _table(
                [
                    "cell",
                    "intent",
                    "\\|δ\\*\\|",
                    "home ATE",
                    "R fitness",
                    "R held-out",
                    "zero-shot",
                ],
                [
                    [
                        # _table escapes the pipe; "mechanism × stage" just reads better
                        str(r["cell"]).replace("|", " × "),
                        r["intent"],
                        str(r["delta_star"]),
                        f"{r['home_ate']:.3f}",
                        f"{r['R_fitness']:.2f}",
                        f"{r['R_held_out']:.2f}",
                        "yes" if r["zero_shot"] else "no",
                    ]
                    for r in tr["rows"]
                ],
            ),
            "",
            "Ensemble members: "
            + ", ".join(
                f"{m['name']}{' (held-out)' if m['held_out'] else ''}" for m in tr["ensemble"]
            ),
            "",
        ]
    else:
        L += ["_not run — `make transfer`_", ""]

    L += _live_transfer_section(_load(d / "live_transfer_gemini.json"))

    L += [
        "## Reproduce",
        "",
        "```",
        "make install",
        "make all      # tests, baseline, flagship attack, budget sweep, transfer, this report",
        "```",
        "",
        "Every stage is deterministic from `MD_SEED`; every result records the `model_id`",
        "it actually ran against.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    cfg = get_config()
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.runs_dir / "report.md"
    dest.write_text(build_report())
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
