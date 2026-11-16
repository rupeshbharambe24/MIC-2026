"""Live-agent failure modes, exercised offline.

Every test here pins a behaviour that, when it broke, produced a *wrong number* rather
than an error -- which is the only reason these paths are worth testing at all.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from minimal_drift.harness.agent import (
    DailyQuotaExhausted,
    EndpointUnreachable,
    OpenAICompatAgent,
    Throttle,
    _is_daily_quota,
)


def _agent(monkeypatch, responses):
    """A compat agent whose HTTP layer replays `responses` (Exception = raise).

    The last entry repeats, so a test can say "this keeps failing" without spelling out
    every retry. Backoff sleeps are stubbed: the retry *policy* is what is under test.
    """
    a = OpenAICompatAgent("test-model", "k", "https://example.test/v1")
    a._throttle = Throttle(rpm=0)
    calls: list[dict] = []
    slept: list[float] = []

    def fake_urlopen(req, timeout=60):
        calls.append(json.loads(req.data))
        r = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("minimal_drift.harness.agent.time.sleep", slept.append)
    a.slept = slept  # type: ignore[attr-defined]
    return a, calls


class _Body:
    def __init__(self, content):
        self._c = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def read(self):
        return self._c

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)  # type: ignore[arg-type]


def test_a_rejected_seed_is_retried_without_it(monkeypatch) -> None:
    """Gemini's OpenAI-compatible endpoint 400s on `seed`, which every request carried.

    Unfixed, every request failed and every member scored as having resisted the attack:
    a transfer rate of zero that looked like a finding.
    """
    a, calls = _agent(monkeypatch, [_http_error(400), _Body('{"ok":1}')])
    assert a._post("prompt", 7) == '{"ok":1}'
    assert "seed" in calls[0] and "seed" not in calls[1]
    assert a.seed_honoured is False  # and the report can say the pairs are noisier


def test_the_seed_is_dropped_only_once(monkeypatch) -> None:
    a, calls = _agent(monkeypatch, [_http_error(400), _Body("a"), _Body("b")])
    a._post("p", 1)
    a._post("p", 2)
    assert "seed" not in calls[-1], "the endpoint's refusal must be remembered"


def test_a_transient_blip_is_retried_then_succeeds(monkeypatch) -> None:
    """A 503 in the middle of a metered run must not throw the run away."""
    a, _ = _agent(monkeypatch, [_http_error(503), _Body('{"ok":1}')])
    assert a._post("p", 1) == '{"ok":1}'
    assert len(a.slept) == 1  # type: ignore[attr-defined]


def test_a_persistent_5xx_eventually_fails_loudly(monkeypatch) -> None:
    a, calls = _agent(monkeypatch, [_http_error(503)])
    with pytest.raises(urllib.error.HTTPError):
        a._post("p", 1)
    assert len(calls) > 1, "a transient status must be retried before giving up"


def test_a_client_error_is_not_retried(monkeypatch) -> None:
    """404 is a configuration mistake; retrying it just burns the quota."""
    a, calls = _agent(monkeypatch, [_http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        a._post("p", 1)
    assert len(calls) == 1


def test_an_unreachable_endpoint_is_never_scored_as_a_resisted_attack(monkeypatch, env) -> None:
    a, _ = _agent(monkeypatch, [_http_error(503)])
    with pytest.raises(EndpointUnreachable):
        a.decide(env.clean_view("intent_ergo_chair", 1), 1)


def test_daily_quota_on_the_compat_path_also_stops_the_run(monkeypatch, env) -> None:
    quota = urllib.error.HTTPError(
        "u", 429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier", {}, None
    )  # type: ignore[arg-type]
    assert _is_daily_quota(str(quota))
    a, calls = _agent(monkeypatch, [quota])
    with pytest.raises(DailyQuotaExhausted):
        a.decide(env.clean_view("intent_ergo_chair", 1), 1)
    assert len(calls) == 1, "a per-day cap must not be retried at all"


def test_throttle_is_a_no_op_when_disabled() -> None:
    t = Throttle(rpm=0)
    t.wait()
    t.wait()  # would sleep 12s at the default rpm


def test_the_demo_does_not_default_to_a_metered_model(monkeypatch) -> None:
    """`make demo` recomputes a real attack per request.

    On "auto" that put gemini behind every page load: slow, rate-limited, and dead the
    moment the daily quota went. Live stays opt-in via MD_DEMO_PROFILE.
    """
    import minimal_drift.web.api as api
    from minimal_drift.harness.agent import SurrogateAgent

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    api._env.cache_clear()
    assert api.DEMO_PROFILE != "auto"
    assert isinstance(api._env().agent, SurrogateAgent)
    api._env.cache_clear()


def test_a_read_timeout_is_retried_like_a_503(monkeypatch) -> None:
    """The gap that cost a 70-minute metered sweep at elite 7 of 8.

    A socket timeout is not an HTTPError and its message carries no status code, so the
    status-string check never matched and the run died on a blip.
    """
    a, _ = _agent(monkeypatch, [TimeoutError("The read operation timed out"), _Body("ok")])
    assert a._post("p", 1) == "ok"
    assert len(a.slept) == 1  # type: ignore[attr-defined]


def test_a_dropped_connection_is_retried(monkeypatch) -> None:
    a, _ = _agent(monkeypatch, [urllib.error.URLError("connection reset"), _Body("ok")])
    assert a._post("p", 1) == "ok"


def test_transient_detection_covers_both_live_agents() -> None:
    from minimal_drift.harness.agent import _is_retryable_status, _is_transient

    assert _is_transient(TimeoutError("timed out"))
    assert _is_transient(urllib.error.URLError("reset"))
    assert _is_transient(ConnectionResetError())
    assert not _is_transient(ValueError("a real bug"))
    # the native Gemini SDK raises typed errors whose text carries the status
    assert _is_retryable_status("503 UNAVAILABLE")
    assert _is_retryable_status("429 RESOURCE_EXHAUSTED")
    assert not _is_retryable_status("400 INVALID_ARGUMENT")
