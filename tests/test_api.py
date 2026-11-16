"""Day 7 gate: the demo serves real, recomputed numbers -- never canned ones."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from minimal_drift.config import get_config
from minimal_drift.web.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "MINIMAL-DRIFT" in r.text
    assert "/api/replay" in r.text  # the UI really talks to the API
    assert "http://" not in r.text.replace("http://www.w3.org", "")  # no external assets


def test_meta_states_its_own_limits(client):
    m = client.get("/api/meta").json()
    cfg = get_config()
    assert m["tau"] == cfg.tau and m["rho"] == cfg.rho and m["seed"] == cfg.seed
    assert m["sign_mode"] in ("hmac-mock", "es256")
    assert len(m["mechanisms"]) == 6 and len(m["stages"]) == 5
    joined = " ".join(m["guardrails"]).lower()
    assert "no real money" in joined and "closed synthetic" in joined
    if not m["live_mode"]:
        assert "surrogate" in joined, "a surrogate run must say so in its own guardrails"


def test_intents_expose_their_drift_targets(client):
    intents = client.get("/api/intents").json()
    assert len(intents) == 10
    flagship = [i for i in intents if i["flagship"]]
    assert len(flagship) == 1 and flagship[0]["id"] == "intent_shoes_10k"
    assert flagship[0]["n_drift_targets"] == len(flagship[0]["targets"]) > 0
    for i in intents:
        assert i["budget_cap"] > 0 and i["currency"] == "INR"


def test_replay_narrates_the_whole_flow_in_order(client):
    r = client.get("/api/replay").json()
    keys = [s["key"] for s in r["steps"]]
    assert keys == [
        "intent",
        "clean",
        "delta",
        "injected",
        "decision",
        "consent",
        "signed",
        "validate",
        "drift",
        "causal",
        "minimal",
    ]
    assert all(s["title"] and s["detail"] for s in r["steps"])
    assert all(s["verdict"] in ("ok", "drift", "blocked", "info") for s in r["steps"])


def test_replay_is_a_valid_payment_with_attributable_drift(client):
    r = client.get("/api/replay").json()
    validate_step = next(s for s in r["steps"] if s["key"] == "validate")
    assert validate_step["data"]["valid"] is True
    assert validate_step["data"]["reasons"] == []
    assert r["attributable"] is True
    assert r["divergence"] > 0 and r["ate"] >= r["tau"] and r["ate"] > r["epsilon"]
    assert r["ci95"][0] > 0


def test_replay_delta_star_is_live_not_stored(client):
    """|δ*| must come from actually running the minimiser."""
    r = client.get("/api/replay").json()
    minimal = next(s for s in r["steps"] if s["key"] == "minimal")
    assert minimal["data"]["keep_calls"] > 0  # the minimiser really ran
    assert r["delta_star"] == minimal["data"]["n_fields_after"] <= r["delta_fields"]
    assert r["cost_star"] <= r["cost"]
    assert len(next(s for s in r["steps"] if s["key"] == "delta")["data"]["ops"]) >= 1


def test_replay_shows_the_merchant_signing_the_truth(client):
    """The signed label must be the real product, not the injected one."""
    r = client.get("/api/replay").json()
    signed = next(s for s in r["steps"] if s["key"] == "signed")["data"]
    injected = next(s for s in r["steps"] if s["key"] == "injected")["data"]
    target = next(o for o in injected["options"] if o["is_target"])
    assert signed["signed_label"] == r["target_name"]
    assert signed["cart_hash"] and len(signed["cart_hash"]) == 64
    assert signed["transaction_data"] and len(signed["transaction_data"]) == 2
    if target["label"] != r["target_name"]:
        assert signed["signed_label"] != target["label"]


def test_replay_accepts_a_specific_target(client):
    intents = client.get("/api/intents").json()
    flagship = next(i for i in intents if i["flagship"])
    sku = flagship["targets"][0]["sku"]
    r = client.get(f"/api/replay?intent={flagship['id']}&target_sku={sku}").json()
    assert r["target_sku"] == sku


def test_replay_rejects_an_unknown_intent(client):
    assert client.get("/api/replay?intent=nope").status_code == 404


def test_replay_reports_an_unattackable_intent_instead_of_faking_one(client):
    """An intent with no in-mandate drift target must 422, not invent a result."""
    intents = client.get("/api/intents").json()
    barren = [i for i in intents if i["n_drift_targets"] == 0]
    for i in barren:
        assert client.get(f"/api/replay?intent={i['id']}").status_code == 422


def test_heatmap_matches_the_declared_grid(client):
    r = client.get("/api/heatmap")
    if r.status_code == 404:
        pytest.skip("no budget sweep on disk; run `make budget`")
    d = r.json()
    assert len(d["cells"]) == 6 * 5
    filled = [c for c in d["cells"] if c["filled"]]
    assert filled, "expected the sweep archive to have elites"
    assert d["summary"]["filled"] == len(filled)
    for c in filled:
        assert c["delta_star"] >= 1 and c["cost"] > 0
        assert 0.0 <= c["R"] <= 1.0 and c["ate"] is not None
    assert d["summary"]["reachable_capacity"] < d["summary"]["capacity"]
    assert d["owasp"]["covered_items"] and d["maestro"]["covered_items"]


def test_missing_measurements_are_reported_as_missing(client, tmp_path):
    """Every panel must 404 rather than render a placeholder number."""
    from minimal_drift import config

    cfg = config.get_config()
    original = cfg.runs_dir
    object.__setattr__(cfg, "runs_dir", tmp_path)
    try:
        for path in (
            "/api/heatmap",
            "/api/budgets",
            "/api/transfer",
            "/api/baseline",
            "/api/defenders",
            "/api/report",
        ):
            assert client.get(path).status_code == 404, path
    finally:
        object.__setattr__(cfg, "runs_dir", original)


def test_measurement_endpoints_serve_what_the_stages_wrote(client):
    for path, key in (
        ("/api/baseline", "mean_epsilon"),
        ("/api/defenders", "in_loop_fpr"),
        ("/api/transfer", "zero_shot_rate"),
    ):
        r = client.get(path)
        if r.status_code == 404:
            pytest.skip(f"{path} not measured yet")
        assert key in r.json()

    r = client.get("/api/report")
    if r.status_code != 404:
        assert "MINIMAL-DRIFT" in r.text and "|δ" in r.text
