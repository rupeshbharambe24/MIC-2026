PY := .venv/bin/python
PYTEST := .venv/bin/pytest

# Which target the experiments run against. Pinned to the surrogate ON PURPOSE: the
# scripts' own default is "auto", which silently goes live whenever a GEMINI_API_KEY
# happens to be present -- so `make all` would stop reproducing the published numbers
# (and start spending quota) on any machine that has a key. Opt in explicitly:
#   make all PROFILE=live
PROFILE ?= surrogate-balanced

.PHONY: install lint fmt type test test-day1 test-day2 test-day3 test-day4 test-day5 test-day6 test-day7 test-blue baseline fpr sweep loop ranking budget transfer blue inflation adaptive latency probes integrity report demo all clean

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests scripts
fmt:
	.venv/bin/black src tests scripts
type:
	.venv/bin/mypy

test:
	$(PYTEST) -q

test-day1:
	$(PYTEST) -q tests/test_schema_parity.py tests/test_divergence.py tests/test_surfaces.py
	$(PY) scripts/clean_run.py --profile $(PROFILE) --intent intent_shoes_10k --seed 1

test-day2:
	$(PYTEST) -q tests/test_validator.py
	$(PY) scripts/baseline_epsilon.py --profile $(PROFILE)

test-day3:
	$(PYTEST) -q tests/test_ate.py
	$(PY) scripts/reproduce_ranking.py --profile $(PROFILE)

test-day4:
	$(PYTEST) -q tests/test_mechanisms.py tests/test_defenders.py

test-blue:
	$(PYTEST) -q tests/test_blue_gate.py
	$(PY) scripts/blue_eval.py --profile $(PROFILE)

blue:
	$(PY) scripts/blue_eval.py --profile $(PROFILE)

inflation:
	$(PY) scripts/minimality_inflation.py --profile $(PROFILE) --tries 30

adaptive:
	$(PY) scripts/blue_adaptive.py --profile $(PROFILE) --budget 400

latency:
	$(PY) scripts/blue_latency.py --profile $(PROFILE)

integrity:
	$(PY) scripts/blue_integrity.py --profile $(PROFILE)

probes:
	$(PY) scripts/protocol_probes.py --profile $(PROFILE)
	$(PY) scripts/defender_fpr.py --profile $(PROFILE)
	$(PY) scripts/mechanism_sweep.py --profile $(PROFILE)

test-day5:
	$(PYTEST) -q tests/test_archive.py tests/test_minimizer.py tests/test_mutator.py
	$(PY) scripts/run_loop.py --profile $(PROFILE) --budget 120 --out data/runs/day5

test-day6:
	$(PYTEST) -q tests/test_transfer.py tests/test_coverage.py tests/test_report.py tests/test_live_agents.py
	$(PY) scripts/run_budget.py --profile $(PROFILE) --budgets 10,25,50,100
	$(PY) scripts/transfer_eval.py --profile $(PROFILE)
	$(PY) -m minimal_drift.metrics.report

test-day7:
	$(PYTEST) -q tests/test_api.py

baseline:
	$(PY) scripts/baseline_epsilon.py --profile $(PROFILE)
fpr:
	$(PY) scripts/defender_fpr.py --profile $(PROFILE)
sweep:
	$(PY) scripts/mechanism_sweep.py --profile $(PROFILE)
loop:
	$(PY) scripts/run_loop.py --profile $(PROFILE) --budget 120 --out data/runs/day5
ranking:
	$(PY) scripts/reproduce_ranking.py --profile $(PROFILE)
budget:
	$(PY) scripts/run_budget.py --profile $(PROFILE) --budgets 10,25,50,100
transfer:
	$(PY) scripts/transfer_eval.py --profile $(PROFILE)
report:
	$(PY) -m minimal_drift.metrics.report

demo:
	.venv/bin/uvicorn minimal_drift.web.api:app --reload --port 8000

all: test baseline ranking budget transfer blue inflation adaptive latency probes integrity report

clean:
	rm -rf data/runs/* .pytest_cache .mypy_cache .ruff_cache
