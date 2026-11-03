# Sieve: plan (v0.2, ambitious edition)

Status: original planning document, kept for history. The measured outcome is in `docs/RESULTS.md`.
Novelty status below was checked against seven primary sources (see `docs/research/notes/novelty_verification.md`).

## Thesis
An inference stack should spend compute and trust in proportion to what a query needs.
Sieve answers locally when the small model is reliably right, escalates when it is not,
and abstains when nothing is reliable, with a **user-specified risk guarantee** and
**measured** cost on consumer hardware.

## The platform: five layers, each shippable alone

| Layer | What it is | Why it matters |
|---|---|---|
| **Sieve Gate** (library) | Backend-agnostic confidence-to-decision engine. Token signals, self-consistency, optional probes. Conformal risk control: "hallucination rate <= alpha on in-distribution traffic". `pip install`-able. | The reusable, upstreamable, citable core. |
| **Sieve Server** | OpenAI-compatible proxy over llama-server / Ollama / vLLM. N-tier cascade (small, large, optional retrieval or remote API), abstain. Streaming-compatible gating. | The thing people run. |
| **Sieve Observe** | Per-decision audit trail (confidence, tier, model + quant, calibration-set hash), Prometheus + OpenTelemetry, calibration-drift alarms, shadow mode, canary and rollback. | Production habits; per-decision auditability matters to enterprise operators. |
| **Sieve Bench** | Open, reproducible benchmark and leaderboard: risk-coverage-cost across models x quantizations x hardware, community-submitted runs. | Benchmarks get cited and starred; this is the adoption engine. |
| **Sieve Lab** | The research: (1) how weight and KV-cache quantization shift the confidence gate, (2) streaming-compatible gating, (3) guarantee behaviour under distribution shift. | The preprint and workshop paper. |

## Flagship result (the headline)
A Pareto frontier of selective accuracy and hallucination rate against latency, VRAM and
joules per query on one consumer GPU, with bootstrap CIs, plus the first measurement of
quantization's effect on gate quality. Target sentence (to be earned, not assumed):
"Matches the large model's accuracy at X% of the energy, with hallucination rate Y%
and a stated guarantee."

## Hard research problems (the real depth)
1. **Streaming vs post-hoc confidence.** Confidence is known only after generation.
   Options: buffered start, early-token probes, abort-and-escalate. Cost of a wasted
   small-model attempt must be in the frontier.
2. **Calibration under quantization and KV-cache quantization.** Apparently unstudied.
3. **Guarantees under shift.** Conformal guarantees assume exchangeability; measure how
   they degrade and detect it (drift alarms).
4. **Co-residency cost.** Two models on 8-24 GB VRAM: swap time, memory pressure.

## Staging and cut order
- **v0.1 (wk 1-6):** Gate + eval harness + first frontier on llama.cpp. Related-work verification.
- **v0.5 (wk 7-10):** Server, OpenAI API, streaming, metrics, Docker, CI on mock backend.
- **v1.0 (wk 11-14):** Observe (audit, drift, shadow/canary), ablations, preprint, blog, launch.
- **Stretch:** Bench leaderboard with community runs; vLLM adapter; retrieval tier.
- **Cut order if behind:** retrieval, vLLM adapter, leaderboard, OpenTelemetry, probes.
  Never cut: Gate, harness, frontier plot, reproducibility.

## Engineering bar
Typed Python 3.11, pytest, ruff, mypy, GitHub Actions on a mock backend (no GPU needed),
pinned dataset and model versions, `make reproduce` for every table and figure, semantic
versioning, changelog, security policy, docs site.

## Upstream plan
Small, honest, disclosed PRs first: llama-server logprobs bugs (#31451, #29456, #31206),
lm-eval-harness calibration metric (#2189). Respect each project's AI-assistance policy
(llama.cpp: AI-written PR text prohibited; vLLM: DCO + disclosure). PRs are written and
submitted by the owner.

## Success signals (outcomes, not promises)
Reproducible repo; frontier plot; preprint; 1+ merged PR; third-party usage. Stars and
workshop acceptance are upside.

## Open decisions
1. Hardware (GPU model/VRAM, or CPU-only).
2. Model pair (default: ~1-3B small, ~8B large, quantized).
3. Name check for "Sieve" (collisions) before launch.

## Novelty status (checked 2026-11-03 against seven primary sources, see docs/research/notes/novelty_verification.md)
Seven papers read via primary sources; all IDs resolve. Do NOT claim as new: calibrated
cascades, abstention in cascades, measured-energy routing, conformal cascades (multiple choice).
- Measured frontier incl. abstention + risk control + latency/VRAM/joules on consumer GPUs: partly open (nobody measures VRAM; nothing with abstention on measured hardware).
- Weight quantization vs gate: partly covered. KV-cache quantization vs gate: open (weak evidence).
- Streaming-compatible gating: open (weak evidence).
- Packaged OpenAI-compatible proxy over llama-server/Ollama: open.
- New closest work: arXiv 2610.23249 (confidence routing on T4s; fails on MMLU-ProX-Lite). Must be engaged directly.
Reframe: measurement study + open tool. Headline = what an abstaining local cascade costs and how
stable the gate is under weight and KV-cache quantization. Probes = offline baseline.
Pending: inspect released repos (UCCI, 2606.02628, 2610.23249), Semantic Scholar/OpenReview scan.
