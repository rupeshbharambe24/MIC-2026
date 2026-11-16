"""The report is the only artefact a human reads, so its table rendering is load-bearing."""

from __future__ import annotations

from minimal_drift.metrics.report import _table, build_report


def _structural_pipes(line: str) -> int:
    """Column separators only -- an escaped `\\|` is content, not a separator."""
    return line.count("|") - line.count("\\|")


def test_a_pipe_in_a_value_does_not_shred_the_table() -> None:
    # archive cell keys are "mechanism|stage" -- unescaped, this silently split a column
    md = _table(["cell", "ate"], [["memory|ranking", "0.750"]])
    for line in md.splitlines():
        assert _structural_pipes(line) == 3, line
    assert "memory\\|ranking" in md


def test_every_row_keeps_the_header_column_count() -> None:
    md = _table(["a", "b", "c"], [["1", "2", "3"], ["x|y", "-", "|"]])
    assert {_structural_pipes(line) for line in md.splitlines()} == {4}


def test_an_already_escaped_pipe_is_not_escaped_twice() -> None:
    # the |delta*| headers are written pre-escaped; double-escaping printed a literal "\\"
    md = _table(["\\|d\\|", "n"], [["1", "2"]])
    assert "\\\\|" not in md
    assert md.splitlines()[0] == "| \\|d\\| | n |"


def test_built_report_has_no_ragged_tables() -> None:
    for block in _table_blocks(build_report()):
        assert len({_structural_pipes(line) for line in block}) == 1, block


def _table_blocks(md: str) -> list[list[str]]:
    blocks, current = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [b for b in blocks if len(b) > 2]


def test_the_suite_is_hermetic() -> None:
    """A real key in .env must not turn unit tests into metered API calls.

    conftest pops the keys after config's load_dotenv has run; get the order wrong and
    every AP2Env built with profile "auto" silently goes live at 12s per call.
    """
    import os

    from minimal_drift.config import get_config

    if os.getenv("MD_TEST_LIVE") == "1":
        return
    assert get_config().live_mode is False
    assert get_config().alt_model_key is None
    assert get_config().heldout_model_key is None


def test_per_day_quota_is_not_retried_like_per_minute() -> None:
    """Both arrive as 429 with a ~60s retryDelay; only one is worth waiting out.

    Treating them alike costs six futile retries per call and dies anyway, so a long
    sweep would stall for hours before reporting a problem that needs a billing change.
    """
    from minimal_drift.harness.agent import _is_daily_quota, _retry_delay

    per_minute = (
        "429 RESOURCE_EXHAUSTED ... 'quotaId': "
        "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier' ... 'retryDelay': '18s'"
    )
    per_day = (
        "429 RESOURCE_EXHAUSTED ... 'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... 'retryDelay': '58s'"
    )
    assert not _is_daily_quota(per_minute)
    assert _is_daily_quota(per_day)
    # the server's own delay is honoured rather than a guessed backoff
    assert _retry_delay(per_minute, 0) == 19.0
    assert _retry_delay("503 unavailable", 3) == 8.0


def test_an_explicit_surrogate_profile_makes_the_whole_process_offline(monkeypatch) -> None:
    """`make all` must reproduce the published numbers on a machine that has a key.

    The target agent is not the only thing that goes live on a key: the reflective
    mutator and the transfer ensemble both do, so gating only the agent left `make all`
    spending quota and crashing two calls deep in the search loop.
    """
    from minimal_drift.config import get_config
    from minimal_drift.harness.agent import GeminiAgent, SurrogateAgent, build_agent

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("ALT_MODEL_API_KEY", "test-key-not-used")
    assert get_config().live_mode is True

    agent = build_agent("surrogate-balanced")
    assert isinstance(agent, SurrogateAgent)
    assert get_config().live_mode is False, "a named surrogate profile must clear the key"
    assert get_config().alt_model_key is None

    # ...and "live" is still honoured, so the gate cannot silently disable the live path
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    assert isinstance(build_agent("live"), GeminiAgent)


def test_live_transfer_says_not_run_when_it_was_not() -> None:
    from minimal_drift.metrics.report import _live_transfer_section

    out = "\n".join(_live_transfer_section(None))
    assert "not run" in out
    assert "cleared" not in out, "an absent measurement must not report a rate"


def test_live_transfer_never_hides_a_partial_low_power_sample() -> None:
    """The section exists to disagree with the headline; it must carry its own caveats."""
    from minimal_drift.metrics.report import _live_transfer_section

    live = {
        "ensemble": [
            {"name": "surrogate-balanced", "vendor": "surrogate", "seed_honoured": True},
            {"name": "gemini:x", "vendor": "google", "seed_honoured": False},
        ],
        "rows": [
            {
                "cell": "memory|memory",
                "delta_star": 1,
                "home_ate": 0.75,
                "members": [
                    {"member": "gemini:x", "ate": 0.25, "attributable": False},
                ],
            }
        ],
        "partial": True,
        "n_elites_available": 8,
        "vendors": ["google"],
        "cross_vendor": False,
    }
    out = "\n".join(_live_transfer_section(live))
    assert "0 of 1" in out
    assert "It does happen" not in out, "nothing cleared the bar; claim nothing"
    assert "Partial" in out and "3 of 8" not in out and "1 of 8" in out
    assert "n.s." in out, "a non-significant effect must not read as a clean zero"
    assert "cross-vendor = **False**" in out
    assert "seed" in out and "gemini:x" in out  # the unseeded caveat names the member
    assert "surrogate-balanced" not in out, "surrogate members are not live members"


def test_live_transfer_names_an_elite_that_drifted_every_model() -> None:
    """The aggregate alone would bury the strongest true statement."""
    from minimal_drift.metrics.report import _live_transfer_section

    live = {
        "ensemble": [
            {"name": "a", "vendor": "google", "seed_honoured": True},
            {"name": "b", "vendor": "google", "seed_honoured": True},
        ],
        "rows": [
            {
                "cell": "omission|retrieval",
                "delta_star": 1,
                "home_ate": 0.75,
                "members": [
                    {"member": "a", "ate": 0.375, "attributable": True},
                    {"member": "b", "ate": 0.5, "attributable": True},
                ],
            },
            {
                "cell": "memory|memory",
                "delta_star": 1,
                "home_ate": 0.75,
                "members": [
                    {"member": "a", "ate": 0.0, "attributable": False},
                    {"member": "b", "ate": 0.0, "attributable": False},
                ],
            },
        ],
        "partial": False,
        "n_seeds": 6,
        "vendors": ["google"],
        "cross_vendor": False,
    }
    out = "\n".join(_live_transfer_section(live))
    lede = out.split("| cell")[0]
    assert "2 of 4" in lede
    assert "omission × retrieval" in lede and "It does happen" in lede
    assert "2 of 4 pairs did not clear" in lede
    # both halves are read off the data; neither mechanism is named in the source
    assert "memory × memory" in lede, "a cell that drifted nothing should be named"
    import inspect

    from minimal_drift.metrics import report

    assert "memory" not in inspect.getsource(report._live_transfer_section), (
        "the report must not hand-write which mechanisms won or lost -- that is the "
        "same defect as the demo's hand-written rationale on day 7"
    )
