# Observability

## Audit trail (`sieve.observe.audit`)
One single-line JSON record per request, written by a bounded queue + writer thread (never blocks the request;
overflow drops and counts, a failing sink is counted and never fails a request). File mode `0600`.

Fields: `ts, request_id, mode, outcome (ok|timeout|cancelled|error), decision (answer|abstain|none), reason,
served_tier, escalated, artifact_id, config_pin, stream, latency_s, prompt_chars, user_hash`, `backend`
(identity of each tier: model id/hash, quant, KV, build, template hash), `cost`, and `tiers[]` with per-tier
`action, reason, signal, score, threshold, signals{...}, defects[], skipped_generation, backend_error, finish`.
In shadow mode `shadow = {action, tier, reason}` is the hypothetical enforce-mode outcome.

**Privacy defaults.** Prompts and responses are not logged. `user` is stored only as a salted HMAC prefix (per
process salt unless `audit.user_hash_salt` is set). Setting `audit.log_content: true` adds `prompt`/`response`
after redaction hooks (`observe/redact.py`: emails, key/bearer-looking strings, 13-19 digit numbers by default;
`redact: false` disables the defaults; add your own with `regex_redactor`). Redaction is regex best effort, not
a privacy guarantee. JSON encoding escapes newlines so prompts cannot forge log lines.

## Metrics (`/metrics`)
All label values come from closed sets (decision, tier 1/2/none, ReasonCode and router reasons, mode, outcome,
fixed route names, HTTP status class); `bounded()` maps anything else to `other`. Prompt text, model names from
requests, user ids and request ids are never labels; a test sends hostile traffic and bounds the series count.
Per-app `CollectorRegistry`, so tests and multiple apps do not collide.

Key series: `sieve_decisions_total{decision,tier,reason,mode}`, `sieve_escalations_total`,
`sieve_request_seconds{decision,outcome}`, `sieve_cost_seconds_total{tier}`, `sieve_wasted_seconds_total`,
`sieve_tokens_total{kind}`, `sieve_score{tier}`, `sieve_signal_defects_total{tier}`,
`sieve_backend_errors_total{tier}`, `sieve_rejected_total{reason}`, `sieve_backend_logprobs_ok{tier}`,
`sieve_ready`, `sieve_escalation_guard_open`, `sieve_drift_*`, `sieve_shadow_would_decide_total`,
`sieve_build_info{artifact_id,config_pin}`.

## Drift alarm (`observe/drift.py`)
Sliding window of tier-1 scores vs a reference sample (calibration-set scores exported by the harness;
`drift.reference_path` `.json` list or `.npy`, optional `reference_sha256`). Uses the Phase A detectors
`calibrate.shift.score_shift_pvalue` (KS) and `psi`. Alarm = KS p-value `< ks_alpha` (default 1e-3) **and**
PSI `> psi_threshold` (default 0.2), checked every `check_every` scores once `min_window` are present. Gauges
`sieve_drift_ks_pvalue`, `sieve_drift_psi`, `sieve_drift_alarm`; one log line on the rising edge. Detection
only: no auto-recalibration or traffic action (D-010). It cannot see a change in P(wrong | score) that leaves
the score histogram unchanged; only graded shadow samples can. Thresholds are untuned engineering defaults;
H5 (>= 80% detection on shifted 200-query windows) is measured by the eval harness, not asserted here.
Without a reference the monitor is disabled.

## Shadow mode
See `docs/server.md`. `observe.summarize_shadow(read_jsonl(path))` aggregates would-answer / would-abstain rates.

## Pinning
`config_pin` = SHA-256 prefix of the secret-free effective settings (API keys and the audit path excluded).
Set `expected_config_pin` / `expected_artifact_id` in deployment manifests to refuse startup when the running
configuration or artifact is not the reviewed one. Both are stamped on every audit record and in
`sieve_build_info`. This replaces canary/rollback automation (D-010). The pin is integrity, not authenticity.

## Dashboard and deployment
`deploy/docker-compose.yml` starts Sieve, Prometheus and Grafana (ports bound to loopback); Grafana is
provisioned with the `deploy/grafana/dashboards/sieve.json` dashboard (readiness, fail-closed reasons,
escalation/abstain rates, latency, wasted work share, score heatmap, drift, rejections, shadow). The compose
file and image were validated with `docker compose config` and YAML/JSON parsing only; they have not been
built or run (no Docker daemon in the build environment).
