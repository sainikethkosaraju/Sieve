# Phase B1 implementation notes (Backend + MLOps + LLM-engineer seats)

Scope: `src/sieve/{backends,router,server,observe}`, tests under `tests/{backends,server,observe}`,
`Dockerfile`, `.dockerignore`, `deploy/`, `docs/server.md`, `docs/observe.md`. Everything below was run
against the mock backend and in-process fake llama-server/Ollama ASGI apps; **no real llama.cpp, GPU, Docker
daemon, or real model was involved**, so nothing here is a measured result about any LLM or about real
llama-server behaviour.

## What was built
* **backends**: `base` (re-exports the `Backend` Protocol, `BackendError`, `BackendHealth`, `SelfTestResult`,
  static URL allow-list `validate_backend_url`), `llamaserver` (async httpx, `/completion` with `n_probs`,
  `/props` identity, `/health`, `n_probs` clamp, faithful response mapping for current and legacy schemas,
  redirects refused, `trust_env=False`), `selftest` (finite/<=0, count, top-k contains chosen, UTF-8 reassembly,
  greedy determinism across repeats), `ollama` (best effort, limits documented in the module docstring),
  `mock` (deterministic, accuracy/confidence knobs, 13 injectable defects, `defect_after`, cancellation counters,
  self-consistency sample behaviour, honours `max_tokens` by truncating).
* **router**: `Router` cascade over the pure `Gate`; `Tier` (backend + server-owned health + slot semaphore);
  preflight skip of tiers with identity mismatch / failed self-test / unreachable (no compute spent, decision
  still produced by the gate); backend exceptions become `Signals` with a defect; self-consistency path
  (greedy + k seeded samples in a TaskGroup); `EscalationGuard`; `CostLedger` (every attempt incl. wasted,
  failed, cancelled and sample attempts); chat-template renderer with versioned hash and special-token defanging;
  shadow mode (serve large, record hypothetical decision).
* **server**: FastAPI app, `/v1/chat/completions` (JSON + SSE), `/v1/models`, `/healthz`, `/readyz`, `/metrics`;
  hand-written strict validation; limits, 429 backpressure, deadline, disconnect watcher; pydantic-settings
  config (YAML/env); `sieve serve` CLI (`sieve.server.cli:main`); maintenance loop that re-tests on identity change.
* **observe**: JSONL audit (bounded queue, drops counted, 0600, privacy by default, redaction hooks), Prometheus
  metrics with closed label sets, drift monitor (KS AND PSI, existing Phase A detectors), shadow summary,
  config/artifact pinning.
* **deploy**: Dockerfile (multi-stage, non-root uid 10001, slim), `.dockerignore`, `deploy/docker-compose.yml`
  (Sieve + Prometheus + Grafana, loopback ports, read-only rootfs), `prometheus.yml`, Grafana provisioning and a
  15-panel dashboard, `sieve.example.yaml`.

## Edits outside my directories (minimal, listed for the Director)
* `pyproject.toml`: `[project.scripts] sieve = "sieve.server.cli:main"`; `server` extra gains `pydantic-settings`,
  `pyyaml`; `dev` extra now includes the server dependencies plus `types-PyYAML` (CI installs only `.[dev]`, and
  the server tests / pydantic mypy plugin need them); `[tool.mypy] plugins = ["pydantic.mypy"]` (without it
  mypy strict reports false `call-arg` errors on pydantic models with defaults).
* No change to Phase A packages or `types.py`. **Incident:** I ran `ruff format src` once (whole tree) and so
  reformatted files of the eval team (`src/sieve/eval/*`) at 07:56:10; formatting-only, but if they report
  unexpected diffs that is why. Their remaining ruff/mypy errors (see below) are not caused by that.

## Deviations from the architect design
1. Gate/Decision shapes follow Phase A (`ANSWER/ESCALATE/ABSTAIN` + tier); the router maps them. Router-only
   reasons `escalation_shed` and `large_unavailable` are strings (not `ReasonCode`) to avoid touching Phase A types.
2. `Backend` has no `stream()`/`health()` in the Protocol (Phase A fixed it as `info` + `generate`); streaming is
   buffered gating at the server, health lives on adapters (`Managed` Protocol, `BackendHealth`).
3. Shed-by-guard requests **abstain** rather than being served by the small tier (serving them ungated would void
   the certificate). The architect's 503 for queue timeout became 429 per this task.
4. Early-abort arm: **not implemented** (documented in `router/core.py` and `docs/server.md`): it needs
   prefix-calibrated thresholds that do not exist.
5. Self-consistency has no load-based degradation (architect suggested capping k by queue depth); the artifact
   fixes the signal, so degrading would change what is certified.
6. Hand-written request validation instead of pydantic body models, to keep error text free of request content.
7. `finish_reason` for abstention is `"sieve_abstain"` by default (non-standard value, configurable).
8. `model_hash` for llama-server is operator-declared (llama-server does not report a weights hash); the optional
   `expected_model_file` check on `/props model_path` poisons the identity when the served file differs.

## Red-team kill tests -> tests
| # | Kill test | Covered by | Not covered |
|---|---|---|---|
| 2 | Fail-closed logprobs | `tests/server/test_failclosed.py` (9 defects x small-only and both-tiers, self-test gating, no compute on untrusted tier); `tests/backends/test_llamaserver.py` (upstream NaN, positive, string, dropped, missing, UTF-8 mapped not repaired); `tests/backends/test_selftest.py` (each check fails on the matching defect; nondeterminism; unreachable) | Behaviour of real llama.cpp bugs #31451/#29456/#31206; the 20x-repeat variance epsilon is a 3-repeat check with eps 1e-3 (untuned) |
| 3 | Manifest mismatch | `test_failclosed.py` (each identity field on both tiers, runtime swap, maintenance loop re-test), `test_api.py::test_artifact_integrity_and_pins_fail_at_startup` (tampered artifact, config/artifact pins), `test_client_sampling_params_are_ignored_not_forwarded` (temperature never reaches the backend); Phase A gate tests for SAMPLING_MISMATCH | Real quant/KV/build values from a live server |
| 7 | SSRF / header injection | `tests/backends/test_urlguard.py` (19 bad URL classes incl. metadata, file://, creds, decimal IP), `test_llamaserver.py` (redirect refused, constructor allow-list, no client headers upstream), `test_load_security.py` (7 override body fields, `model: http://169.254...`, 7 override headers, request-id CRLF, Authorization/Cookie not forwarded, only fixed upstream paths) | DNS rebinding of an allow-listed hostname |
| 8 | Canary privacy | `test_load_security.py::test_canaries_absent_...` (canary prompt, API key, user id, newline log-forging across audit file, `/metrics`, caplog at DEBUG, all response bodies incl. error paths), `test_metric_labels_are_bounded_...`, `test_content_logging_is_opt_in_and_redacted`, `test_unhandled_error_does_not_leak`; `tests/observe/test_observe.py` | Crash dumps, uvicorn access log of URLs, traces (none exist) |
| 9 | Cancellation & load | `test_load_security.py`: disconnect cancels backend (ASGI-level), 40 simultaneous aborted requests leave no in-flight work or limiter slots, cancel during escalation handoff, deadline 504 + cancel, 429 (full queue and queue timeout), 50 concurrent mixed (only 200/429), oversize (content-length and chunked) 413, malformed JSON x9 and 18 validation cases never 5xx, slow body 408 | `/slots` orphan check on a real llama-server; 100 (not 40) aborts; memory bound under load; header-level slowloris (uvicorn/reverse proxy); over-context in tokens |
| 12 | Adversarial confidence (server side) | `test_api.py::test_max_tokens_one_...` (truncation fails closed), `logit_bias`/`response_format`/`logprobs`/`tools` rejected (validation table), client temperature/top_p/seed/stop ignored or rejected, special-token role forging defanged | Answered-wrong rate under injected prompts (eval harness) |
| - | Escalation amplification | `test_router.py::test_guard_*` (window invariant, shed requests abstain, no large-tier compute for shed) | Cap value is untuned |
| - | Drift alarm | `tests/observe/test_observe.py` (10 stable streams: 0 false alarms; shifted stream fires < 300 obs), `test_drift_metrics.py` (end to end through the server) | H5 power claim (>= 80% on shifted windows) is for the eval harness |
Kill tests 1, 4, 5, 6, 10, 11 are statistical/eval tests owned elsewhere.

## Verification (exact outputs are in the final report)
Commands: `ruff check`, `ruff format --check`, `mypy`, `pytest` over the whole repo, plus a real-uvicorn smoke
(`python -m sieve.server.cli serve` with mock backends: `/readyz`, SSE, JSON, `/metrics`, audit file).
Entry point `sieve` is declared but the console script was not installed into the venv; the module form was run.

## Known gaps / risks
* Docker image and compose stack were never built/run (no daemon); `docker compose config` and YAML/JSON parsing
  pass. llama-server and Prometheus/Grafana images pinned by tag, not digest.
* The Phase A `validate_completion` does not check "chosen token is inside top-k" (the self-test does, once at
  startup/identity change); per-request a chosen-only score therefore trusts the top list's absence/presence.
* Mock timings are artificial; cost seconds in tests are wall clock with `asyncio.sleep`.
* Disconnect propagation relies on polling `request.is_disconnected()` every 50 ms; cancellation latency is
  bounded by that plus httpx socket close. Verified at ASGI level only (httpx's ASGITransport cannot send a
  disconnect), not against a real TCP client + real llama-server slot release.
* The audit queue thread is per sink; no rotation. Audit `latency_s` is measured to the decision, not to the
  last streamed byte.
* Tests use `pytest.mark.anyio` (the anyio plugin ships with FastAPI/Starlette deps; no pytest-asyncio needed);
  each test dir carries its own `anyio_backend` fixture so the dirs are independent of a shared conftest.
  Timing-based tests (429, deadline, drift stream) use generous margins but are not hermetic under extreme CPU load.
* Python 3.11 not available locally (ran on 3.13); code uses only 3.11 features (`asyncio.timeout`,
  `TaskGroup`, `StrEnum`), CI covers 3.11/3.12.
* Drift thresholds (KS 1e-3 AND PSI 0.2) and guard rate 0.8 are untuned defaults and should be frozen or
  recorded as deviations before any headline claim about them.
