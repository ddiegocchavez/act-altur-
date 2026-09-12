PYTHON ?= python3
PY := .venv/bin/python
WORKERS ?= 6
PUBLIC_URL ?=

.PHONY: all install data phase1 phase2-local phase2-docker phase2-public phase3 audit robust audio-robust stress serve

# Sequential recursive calls keep the gates ordered, even with make -j.
all:
	$(MAKE) install
	$(MAKE) data
	$(MAKE) phase1
	$(MAKE) phase2-local
	$(MAKE) phase2-public
	$(MAKE) phase3
	$(MAKE) stress
	@echo "Public baseline, behavior ablations and latency stress reproduced. Docker deferred."

install:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) prepare_data.py

phase1:
	$(PY) -u phase1.py --workers $(WORKERS)
	$(PY) write_results.py

phase2-local:
	$(PY) -c 'import json; assert json.load(open("reports/phase1.json"))["gate"]["passed"], "Phase 1 gate failed"'
	$(PY) -u verify_local.py --model artifacts/baseline_model.joblib
	$(PY) write_results.py

phase2-docker:
	$(PY) -u verify_docker.py
	$(PY) write_results.py

phase2-public:
	@test -n "$(PUBLIC_URL)" || (echo "Phase 2 blocked: set PUBLIC_URL to the deployed HTTPS endpoint"; exit 2)
	$(PY) -u evaluate_http.py --url "$(PUBLIC_URL)" --allow-remote --output reports/phase2_public_http.json
	$(PY) write_results.py

phase3:
	$(PY) -m unittest -v test_behavior
	$(PY) -u phase3.py --workers $(WORKERS)
	$(PY) -u verify_selected.py
	$(PY) write_results.py

audit:
	$(PY) -u audit_shortcuts.py

robust:
	$(PY) -m unittest -v test_behavior test_robustness
	$(PY) -u phase4_robust.py --workers $(WORKERS)

audio-robust:
	$(PY) -u audio_robustness.py

stress:
	$(PY) -u stress_latency.py
	$(PY) write_results.py

serve:
	$(PY) serve.py
