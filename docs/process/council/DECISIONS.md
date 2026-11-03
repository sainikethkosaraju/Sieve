# Decision record

| ID | Date | Decision | Owner-approved | Rationale / evidence |
|---|---|---|---|---|
| D-001 | 2026-11-03 | Charter: 15 permanent seats, 8-step loop, independent Release Auditor | yes (owner request) | docs/council/CHARTER.md |
| D-002 | 2026-11-03 | Work on `claude/great-babbage-a6srla`; after Release Auditor verdict, merge into `main` and push directly (no PR) | yes | owner answer |
| D-003 | 2026-11-03 | Definition of done: v1.0 software (Gate, Server, Observe, harness, CI, docs) + one real measured T4 baseline, honest incl. negative results. No paper/blog this round. | yes | owner answer |
| D-004 | 2026-11-03 | Compute: Kaggle 2x Tesla T4 via REST (`www.kaggle.com/api/v1`) with proxy-injected bearer secret; smoke test passed | yes | smoke kernel `ksainiketh/sieve-smoke` |
| D-005 | 2026-11-03 | Novelty reframed to measurement study + open tool; KV-cache-quantization vs gate is the lead angle | pending owner awareness | research_notes/.../novelty_verification.md |

## Unresolved risks
- Novelty evidence is limited (arXiv API/listing unreachable); released repos of UCCI, 2606.02628, 2610.23249 not yet inspected.
- Kaggle weekly GPU quota is unknown; runs must be resumable.
- llama.cpp CUDA build and GGUF load on T4 not yet proven.
- Confidence routing may fail on knowledge/reasoning tasks (see arXiv 2610.23249); negative results must be reported.

## Consolidated after independent design reviews (2026-11-03)
Reviews: reviews/architect-design.md, reviews/research-ml-design.md, reviews/redteam-design.md (written independently, no cross-exposure).
| ID | Decision | Source |
|---|---|---|
| D-006 | Architecture per architect-design.md: pure `Gate(signals, artifact)` shared by server, eval harness, shadow mode; only backends do network IO; single-process asyncio; buffered gating default, early-abort an opt-in measured arm. | Architect |
| D-007 | Guarantee claim limited to: with prob >= 1-delta over the calibration draw, grader-judged error among answered queries <= alpha, for exchangeable traffic and a frozen model/quant/prompt/sampler. Thresholds via fixed-sequence Learn-then-Test on full-cascade loss. Never claim per-query or factual truth. | ML/Researcher |
| D-008 | Gate fails closed on logprob defects, sampling mismatch, and calibration-artifact mismatch (model/quant/KV/build/template hash). | Red-Team |
| D-009 | Red-Team's 12 kill tests are release gates; tests on guarantee coverage, fail-closed, calibration mismatch, non-inferiority block the headline claim; SSRF/header injection, log privacy, cancellation/load block any server release. | Red-Team |
| D-010 | Cut for v1.0: OpenTelemetry, vLLM adapter, retrieval tier, hidden-state probes, leaderboard, N-tier generality, learned calibrators, multi-worker, web UI. Canary/rollback reduced to shadow mode + config version pin. | Architect/Product |
| D-011 | Headline experiment protocol per research-ml-design.md (TriviaQA 2000/2000/2000 + shift sets, 50 re-splits, paired bootstrap, pre-registered hypotheses H1-H7, negative results published). Pre-registration committed BEFORE the T4 run. | ML/Researcher |
| D-012 | Implementation agents edit only their assigned directories and do not commit; Director commits. Implementer never sole reviewer; QA, Red-Team and Release Auditor review independently. | Charter |

## Evidence: real T4 smoke test (2026-11-03) results/evidence/t4_llamacpp_smoke_2026-11-03.json
| ID | Finding |
|---|---|
| D-013 | llama.cpp @ fac3a89 builds with CUDA 13.0 for sm_75 on Kaggle ONLY with `-DGGML_CUDA_NO_VMM=ON` (the CUDA-driver-stub path attempt failed). Build took ~29 min; reuse the compiled `llama-server` via a Kaggle kernel data source. |
| D-014 | llama-server returns per-token logprobs with top-k on `/completion` (`n_probs`) and `/v1/chat/completions`; greedy output was deterministic across two runs; quantized KV (`-fa on -ctk q8_0 -ctv q8_0`) works on T4. Single-run speeds: 1.5B Q4_K_M ~129.5 tok/s (default) / ~123.5 (KV q8_0); 7B Q4_K_M ~48.6 tok/s. NOT benchmarks. |
| D-015 | Kaggle output files are served from kaggleusercontent.com (blocked in the sandbox); results return through the kernel log. Large artefacts need that host allowed. |
| D-016 | Raw `/completion` prompts make the 7B open with " The" (logprob -0.25) before " Paris": answer-span extraction / answer prefix is mandatory, as the ML review warned. |
| D-017 | Repo appears PUBLIC (github.com page returns 200 unauthenticated). Everything pushed, including unverified research notes, is visible. Owner to confirm intent. |

## Owner decision (2026-11-03)
| ID | Decision |
|---|---|
| D-018 | Repo stays PUBLIC on purpose: recruiters are an intended audience (resolves D-017). Release gate added: presentation quality. Before merge to main: README leads with what is proven vs not proven (no unmeasured claims); unverified research notes moved under `docs/research/` with an explicit "unverified, snippet-based" banner or removed; no secrets, tokens, personal data or Kaggle output URLs in history; commit authorship checked; AI assistance disclosed honestly in README and commit trailers (also required by llama.cpp/vLLM policies for any upstream PR). |

## Audit triage (2026-11-03)
Audits: reviews/governance-audit.md, eval-audit.md, redteam-code-audit.md.
| ID | Decision |
|---|---|
| D-019 | Served prompt uses the SAME template as calibration, applied to the last user message (one shared prompt builder; template hash identical in eval and server). The certificate covers traffic of that task type only; docs say so. Fixes F-01. |
| D-020 | Small tier switches from Qwen2.5-3B (Qwen research licence, non-commercial) to Qwen2.5-1.5B-Instruct (Apache-2.0), before any real experiment; recorded as a dated Deviation. |
| D-021 | B-once / C-hash is documented as a DETECTION control (tripwire), hardened (recomputed hashes, atomic ledger, once-only C, code and config recorded), not as an access-control guarantee. |
| D-022 | No real headline run until: real-server canary passes (>=95% finish==stop per tier, tokens join to text), F-01/F-02/F-03 fixed, eval deviations recorded, H3/H6/H1 verdict logic fixed. |
| D-023 | Release gate: `results/mock` regenerated and labelled SYNTHETIC in every artefact; every public claim traceable to evidence. |
