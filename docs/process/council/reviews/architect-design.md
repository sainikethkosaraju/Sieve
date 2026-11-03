# Sieve v1.0 architecture: Principal Systems Architect (independent opinion)

Basis: PLAN.md v0.2, landscape report, novelty_verification.md. No code exists to inspect; all claims below are design, not measured. Frame: the product is a *measurement study + thin tool*. Every module exists to make the frontier experiment reproducible or the proxy honest. One process, stdlib + httpx + FastAPI/uvicorn + numpy. No LangChain, no ORM, no OTel in v1.

## 1. Package layout and public interfaces

```
src/sieve/
  types.py        # frozen dataclasses: Request, Generation, TokenLP, Signals, Decision
  backends/       # base.py (Protocol), llamaserver.py, ollama.py, mock.py
  signals/        # logprob.py (pure fns), consistency.py
  calibrate/      # crc.py (conformal risk control), isotonic.py, store.py (JSON artifact)
  gate.py         # Gate: Signals -> Decision, using a CalibrationArtifact
  router.py       # Router: orchestrates tiers, no HTTP knowledge
  server/         # app.py (FastAPI, /v1/chat/completions, /v1/models, /metrics, /healthz), stream.py
  observe/        # audit.py (JSONL sink), metrics.py (prometheus_client), drift.py, shadow.py
  eval/           # runner.py, cache.py (sqlite/jsonl by item-key), datasets.py, report.py
  cli.py          # sieve serve | calibrate | eval | report
```
Dependency rule: `types <- signals/calibrate/gate <- router <- server`; `eval` imports router/gate but never `server`. Backends are the only code that does network IO.

```python
@dataclass(frozen=True)
class TokenLP: token: str; logprob: float; top: tuple[tuple[str,float],...]  # top-k alternatives
@dataclass(frozen=True)
class Generation:
    text: str; tokens: tuple[TokenLP,...]|None; finish: Literal["stop","length","error"]
    t_first_token_s: float; t_total_s: float; n_prompt: int; n_gen: int
    logprobs_ok: bool            # backend health check verdict, see decision #3

class Backend(Protocol):
    name: str; model_id: str; quant: str; kv_quant: str          # identity used as calibration key
    async def generate(self, messages: Sequence[Message], p: SamplingParams, *, want_logprobs: bool) -> Generation: ...
    def stream(self, messages: Sequence[Message], p: SamplingParams, *, want_logprobs: bool) -> AsyncIterator[Chunk]: ...
    async def health(self) -> BackendHealth: ...                 # includes logprob self-test result
    async def aclose(self) -> None: ...

@dataclass(frozen=True)
class Signals:           # all floats are model-own; no learned features in v1
    mean_lp: float; min_lp: float; first_tok_margin: float; seq_entropy: float
    agreement: float|None   # self-consistency, None if not sampled
    n_samples: int

class Signal(Protocol):  # a pure function wrapper, name for audit
    name: str
    def __call__(self, gens: Sequence[Generation]) -> float: ...

@dataclass(frozen=True)
class CalibrationArtifact:
    model_id: str; quant: str; kv_quant: str; signal: str
    lam: float; alpha: float; delta: float; n_cal: int; cal_hash: str; created: str
    abstain_lam: float|None   # second threshold for tier 2

class Calibrator(Protocol):
    def fit(self, scores: np.ndarray, correct: np.ndarray, *, alpha: float, delta: float) -> CalibrationArtifact: ...

Action = Literal["answer_small","escalate","answer_large","abstain"]
@dataclass(frozen=True)
class Decision:
    action: Action; tier: int; score: float; threshold: float; artifact_id: str
    reason: Literal["conf_ok","low_conf","logprobs_unavailable","backend_error","large_low_conf"]

class Gate:
    def decide(self, sig: Signals, art: CalibrationArtifact, tier: int) -> Decision: ...

class Router:
    async def run(self, req: Request) -> RouterResult: ...    # buffered path
    def run_stream(self, req: Request) -> AsyncIterator[Event]: ...

class AuditSink(Protocol):
    def write(self, rec: AuditRecord) -> None: ...            # non-blocking; bounded queue, drops counted
    def close(self) -> None: ...
```
Gate is a pure function of (signals, artifact): this makes the eval harness, the server, and shadow mode share *one* decision path, which is the correctness property that matters.

## 2. Streaming, honestly

Confidence exists only after tokens exist. Options: (a) buffered-then-gate, (b) early-abort probe (gate on running min/mean logprob after K tokens, abort and escalate), (c) stream-speculatively-then-retract (impossible over OpenAI SSE; clients have already rendered).

**Pick: buffered by default; early-abort as an opt-in `X-Sieve-Early-Abort` mode measured in the experiment, never default.** Buffered means TTFT = full small-model generation time; the SSE stream replays the final answer's text (from small or large) after the decision. Declare this in docs and expose `sieve_ttft_seconds` split by action. Cost model, written into the frontier: wasted-attempt cost = small-model latency + energy for every escalated/abstained query; buffered adds no extra waste but adds TTFT = T_small (+ T_large on escalate). Early-abort: gate checked every token after warmup K (e.g. 8-16) on running min logprob against a *separately calibrated* prefix threshold (calibration must be done on prefixes, not full answers, or the guarantee is void); on trigger, cancel the small request (llama-server closes slot on disconnect), start large. It cuts waste on escalations but inflates false aborts, and the CRC guarantee applies to the prefix rule only, not the final one. Early-abort that *did not* abort still needs a final gate before the stream closes, but tokens are already sent, so abstain/escalate after the fact cannot be honored: therefore early-abort mode only protects against escalation, and abstention is only possible pre-first-byte. State this limit. Client disconnects cancel upstream via task cancellation in both modes.

## 3. Concurrency, state, VRAM

- Single uvicorn worker, asyncio, one `httpx.AsyncClient` per backend. Immutable state: loaded `CalibrationArtifact`s, config. Mutable state: metrics registry, audit queue, drift window (deque + lock-free single-writer). No DB in the request path. Multi-worker is out of scope (Prometheus multiproc and drift state are not worth it).
- Backpressure: per-backend `asyncio.Semaphore(slots)` matched to `llama-server --parallel`; queue timeout returns 503 with Retry-After. Self-consistency samples use `n` parallel requests in the same semaphore, so a query with k samples costs k slots; Router caps k by current queue depth (degrade to logprob-only under load, audited as such).
- Failure isolation: small-tier error -> escalate (reason `backend_error`); large-tier error -> abstain/503 per config; audit sink failure never fails a request.
- VRAM (T4 = 15 GB usable each; no FP8; llama.cpp GGUF K-quants run fine): **default plan, two llama-server processes, small on GPU0, large on GPU1** (e.g. 3B Q4_K_M ~2.5 GB, 8B Q4_K_M ~5-6 GB plus KV). Zero swap, so measured latency isolates the gate. **Single-T4 co-residency** (both resident, small ~3 GB + large ~6 GB + KV for each, `-c` capped) is the consumer-relevant config and a second experimental arm; record `nvidia-smi` peak per arm. **Swap arm** (load large on demand) measured only offline with cold-load times; not a serving mode in v1. Never rely on Ollama keep-alive eviction as a design; set `OLLAMA_MAX_LOADED_MODELS` and report it. Weight/KV quant sweeps launch fresh servers per configuration from the harness (`--cache-type-k/v`), keyed into the calibration artifact.

## 4. Five riskiest decisions

1. **Gating point: buffered vs early-abort.** Alts: early-abort default; retract. Pick buffered default (above): only honest option supporting abstain and a valid guarantee; early-abort is a measured extension. Risk: TTFT penalty reviewers dislike; mitigated by reporting it.
2. **Guarantee target and calibration unit.** Alts: marginal accuracy over answered (CC-style union bound), per-tier independent CRC, end-to-end risk on the final answered set. Pick: calibrate **end-to-end selective risk** (error rate among non-abstained outputs of the *composed* system) with one scalar knob per tier searched on a grid with Hoeffding/binomial upper bound; artifact keyed (model, quant, kv_quant, signal, prompt template). Open-ended tasks need a correctness label: v1 restricts claims to tasks with automatic grading (MCQ + exact-match short answer) and says so. Risk: composed guarantee is the hard bit; fall back to per-tier bounds with union bound if the joint search is unsound.
3. **Trusting llama-server logprobs (known bugs #31451, #29456, #31206).** Alts: trust; reimplement via `/completion` `n_probs`; compute own via llama-cpp-python. Pick: use `/completion` (not chat) with `n_probs`, plus a **startup self-test** (fixed prompt, check probabilities sum <=1, monotone top-k, finite, matches a pinned golden within tolerance) setting `logprobs_ok`; if false the gate refuses to use logprob signals (reason `logprobs_unavailable`; escalate or consistency-only). Pin llama.cpp commit in the repo and record in every artifact. Risk: tokenization/template drift between chat route and `/completion`; render the chat template ourselves once, version it.
4. **Eval harness cache design.** Alts: cache final metrics; cache raw generations keyed by prompt only. Pick: cache every *Generation incl. per-token top-k* keyed by `hash(dataset_item, model_id, quant, kv_quant, sampling, template_ver, llama_cpp_commit, seed, sample_idx)` in append-only JSONL shards + manifest; calibration, gate, and all plots recompute from cache on CPU. Kaggle resumability = skip keys present; flush every N items; output dataset versioned. Risk: size (top-k for every token); store top-5, float16, gzip. This also lets CI run the identical pipeline against a checked-in tiny fixture cache.
5. **Cost accounting.** Alts: model cost as constants; estimate from FLOPs. Pick: record wall time per tier from the actual run, and for joules use NVML polling (50-200 ms, idle subtracted) around each *experiment cell* not each request on shared GPUs; in the frontier, cost of a query = measured small cost + (escalated ? large cost : 0) taken from cached per-item timings, so the frontier is recomputable from cache without re-running GPUs. Risk: per-item timings measured serially (batch=1) won't match concurrent serving; label as such and run one concurrency-load arm.

## 5. Cut for simplicity

- OpenTelemetry (already first in cut order), vLLM adapter, retrieval tier, hidden-state probes (offline baseline script only, outside package), leaderboard/Bench, canary/rollback automation (keep shadow mode only: compute the decision, log it, always serve the large/baseline answer).
- N-tier generality: hardcode 2 tiers + abstain behind a `tiers: list` config shape that validates len==2.
- Learned calibrators (isotonic yes as an *ablation* in eval; CRC threshold is the product). No Platt/HCMA variants.
- Multi-worker, auth, rate-limiting beyond semaphores, hot reload of artifacts (restart instead), a DB, a web UI. Drift alarm = rolling KS/PSI on the score histogram vs calibration set, one gauge + log line; no auto-recalibration.
- Tool calls/function-calling/multimodal passthrough: reject with 400, documented.

## Summary
1. Pure `Gate(signals, artifact)` shared by server, harness and shadow; backends are the only IO.
2. Buffered streaming by default (TTFT = full small generation); early-abort is an opt-in measured arm with prefix-calibrated thresholds; abstain only pre-first-byte.
3. Single asyncio process; two llama-server processes (small GPU0, large GPU1) as default, single-T4 co-residency and swap as experimental arms.
4. Riskiest: gating point, composed-risk guarantee, llama-server logprob trust (startup self-test, `/completion`), cache-everything harness, measured-cost accounting.
5. Calibration artifacts keyed by (model, quant, kv_quant, signal, template, llama.cpp commit).
6. Cut: OTel, vLLM, retrieval, probes, leaderboard, canary, N-tier, learned routers, multiprocess.
