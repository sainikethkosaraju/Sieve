# Sieve server

OpenAI-compatible proxy that runs the cascade **small model -> gate -> (large model -> gate) -> answer / abstain**.
Every answer/escalate/abstain outcome comes from the pure `sieve.gate.Gate` and a hash-verified
`CalibrationArtifact`; the router and server contain no decision logic of their own.

```
sieve serve --config deploy/sieve.example.yaml          # entry point added to pyproject.toml
python -m sieve.server.cli serve --config ...           # equivalent
```

Install: `pip install -e ".[server]"` (the `dev` extra includes the server dependencies).

## What the guarantee covers (and does not)
The certificate (D-007) holds only for the frozen configuration recorded in the artifact: model hash,
quantization, KV type, llama.cpp build, rendered prompt template, decoding parameters, signal and grader.
The server therefore enforces that configuration instead of trusting callers:

* Identity of each backend (`BackendInfo`) is compared with the artifact on **every** request (the gate does
  it); a mismatch never answers. The mismatching tier is not even called (no wasted GPU time).
* Decoding parameters come from the artifact. Client `temperature`, `top_p`, `seed`, `stop`, penalties are
  ignored (listed in `sieve.ignored_params`) or, with `client_sampling: reject`, rejected with 400.
  `logit_bias`, `response_format`, grammar, `logprobs`, `tools`, `n>1` are always rejected.
* A client `max_tokens` is honoured but truncation (`finish_reason == "length"`) is a defect that fails
  closed, so `max_tokens=1` cannot inflate confidence.
* Client text is defanged for chat-template special tokens (`<|` becomes `< |`) so roles cannot be forged.
* Traffic must be exchangeable with the calibration data; prompts that differ from the calibration prompt
  format (e.g. client system prompts) are outside the certificate. Use the drift alarm and shadow mode.

## Endpoints
| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | non-stream and SSE; `sieve` extension block; `X-Sieve-Decision/Tier/Artifact` headers |
| `GET /v1/models` | configured `model_name` and aliases |
| `GET /healthz` | liveness, always 200 |
| `GET /readyz` | 200 only if every tier is reachable, passed the logprob self-test and matches the artifact; else 503 with details |
| `GET /metrics` | Prometheus; requires the API key unless `metrics_public: true` |

Only these paths are valid; docs/openapi routes are disabled. All errors use the OpenAI envelope
`{"error": {"message","type","param","code"}}`; messages are static and never echo request content.

### Response extension
```json
"sieve": {"decision": "answer|abstain", "tier": 1|2|null, "reason": "confident|low_confidence|logprob_defect|identity_mismatch|sampling_mismatch|signal_unavailable|tier_not_certified|escalation_shed|large_unavailable",
          "escalated": true, "mode": "enforce|shadow", "artifact_id": "...", "score": -0.02, "threshold": -0.5,
          "cost": {"total_seconds": 0.0, "wasted_seconds": 0.0, "total_tokens": 0, "wasted_tokens": 0, "attempts": 0},
          "ignored_params": []}
```
`usage` reports **billed** tokens (all attempts, including a wasted small attempt), not just the served answer.

### Abstention
HTTP 200 with a well-formed assistant message (`abstain_message`) and `finish_reason: "sieve_abstain"`
(configurable to `content_filter` or `stop` for strict clients). Clients should key on `sieve.decision` or the
`X-Sieve-Decision` header. The backend's low-confidence text is never returned.

### Streaming = buffered gating
Confidence exists only after tokens exist, so the server runs the whole cascade first and then replays the
final text as SSE chunks. Consequences: time-to-first-token equals the full generation time of the tier(s)
used, abstention is still possible before the first byte, and the decision is in the first chunk and headers.
**Early-abort (gating a running prefix) is not implemented**: it requires thresholds calibrated on prefixes,
which do not exist (a rule calibrated on full answers has no guarantee on prefixes).

## Backends
`backends/llamaserver.py` uses `/completion` with `n_probs`, `/props` (build, model file) and `/health`.
Known llama.cpp logprob bugs are handled by mapping responses faithfully (never repairing) so that
`validate_completion` flags them, by clamping `n_probs` client-side (<= 64, bug #29456), and by the **startup
self-test**: finite and `<= 0` logprobs, one entry per generated token, chosen token inside its top-k list,
sorted top-k with mass `<= 1`, UTF-8 pieces reassemble, greedy repeats identical within `eps`. A failing
self-test sets `logprobs_ok=False`: the tier's logprob signals are not used and `/readyz` returns 503. A
maintenance loop (`health_interval_s`) re-reads `/props` and re-runs the self-test when the identity changes.
`backends/ollama.py` is best effort (see its docstring: logprob support is version dependent, no build or
KV-type identity). `backends/mock.py` is the deterministic CI backend with injectable defects.

## Security controls
* **No client-controlled upstreams.** URLs exist only in config and are validated against a static host
  allow-list (`allowed_backend_hosts`); http(s) only, no credentials, query or fragment; link-local,
  unspecified, multicast and cloud-metadata hosts are always refused; redirects are never followed;
  `trust_env=False` so proxy env vars are ignored. Unknown body fields (`base_url`, `backend`, ...) and
  `X-Backend`-style headers are rejected with 400. Hostnames are not resolved, so DNS rebinding of an
  allow-listed name is the operator's responsibility.
* **Nothing from the caller is forwarded upstream**: backends build their own requests; client
  `Authorization`/`Cookie`/forwarding headers cannot reach them. `X-Request-ID` is accepted only if it matches
  `[A-Za-z0-9._-]{1,64}`, otherwise replaced.
* **Auth.** Bearer API keys (constant-time compare). Binding a non-loopback host without `api_keys` is a
  configuration error. TLS is not terminated here; use a reverse proxy.
* **Limits.** Body cap enforced on `Content-Length` and while streaming chunked bodies (413), body read timeout
  (408), message/prompt/`max_tokens` caps, strict JSON (duplicate keys, NaN/Infinity, deep nesting -> 400),
  `Content-Type` must be JSON (415).
* **Backpressure.** At most `max_concurrency` requests run, `max_queue` wait up to `queue_timeout_s`; beyond
  that 429 with `Retry-After`. Each tier has its own slot semaphore (self-consistency samples consume slots).
* **Deadline and cancellation.** `request_deadline_s` -> 504. A client disconnect cancels the cascade task, which
  cancels the in-flight `httpx` call and so releases the llama-server slot. Time burnt by cancelled attempts is
  still accounted.
* **Escalation guard.** Sliding window of `guard.window` requests; at most `max_escalation_rate * window`
  escalations are granted. A shed request is **abstained** (`escalation_shed`), never answered from the small
  tier, so the cap bounds worst-case cost (2x) under adversarial hard prompts without breaking the guarantee.
  The default 0.8 is an untuned safeguard; set it from your measured escalation rate. `1.0` disables it.
* Unhandled exceptions return a generic 500 and log only the exception type.

## Fail-closed summary
Tier 1 failing closed escalates; tier 2 failing closed abstains. Triggers: logprob defects (NaN, positive,
missing or mismatched count, UTF-8 gap, truncation, error finish, greedy mismatch), backend errors/timeouts,
failed self-test, identity or sampling mismatch, unavailable score, escalation shed, no large tier.

## Cost accounting
Each backend call is an attempt (`router/cost.py`), including the wasted small attempt, self-consistency
samples, failed and cancelled calls. `wasted_*` is everything that is not the served tier's primary generation
(all of it on abstain). Seconds are wall clock around the call; batch-1 numbers are not concurrent-serving
numbers.

## Self-consistency
If a tier's artifact signal is `agreement` or `neg_cluster_entropy`, the router fires the greedy answer plus
`n_samples` sampled generations (seeds `seed+i`) in parallel and gates on the cluster statistic. Samples cost
tier slots and are counted as overhead. The served text is the greedy answer; a truncated greedy answer still
fails closed. There is no load-based degradation to logprob-only (deliberate: the artifact fixes the signal).

## Shadow mode
`mode: shadow` always serves the large (baseline) answer and records what the gate **would** have done
(`sieve.shadow`, the audit record and `sieve_shadow_would_decide_total`). Use it to canary an artifact on real
traffic before enforcing. If the baseline tier fails the request fails with 502 (there is no safe fallback).

## Configuration
`ServerSettings` (pydantic-settings, `extra="forbid"`): YAML via `from_yaml`, environment overrides with
`SIEVE_` prefix and `__` nesting (`SIEVE_LIMITS__MAX_CONCURRENCY=4`); see `deploy/sieve.example.yaml`.
`expected_artifact_id` / `expected_config_pin` refuse startup on mismatch (see `docs/observe.md`). A corrupt or
tampered artifact (hash mismatch) also refuses startup.

## Known limits
Single process, single worker (D-006/D-010). Prompt size is bounded in characters, not tokens (the backend
rejects over-context requests, which fails closed as a backend error). Header-level slowloris is NOT
mitigated by Sieve or uvicorn: `timeout_keep_alive` only applies after a response, and no timeout fires on a
connection that sends a partial request line (red-team F-10). `sieve serve` sets `limit_concurrency` (requests beyond
it get 503) and `h11_max_incomplete_event_size`, but anything not on loopback MUST sit behind a reverse proxy that
enforces header-read and idle timeouts.
No hot reload of artifacts: restart. The audit sink does not rotate files; use logrotate `copytruncate`.
Uvicorn's access log prints request paths (not bodies); do not put secrets in URLs.
