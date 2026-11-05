# Roadmap

What is done, what is next, and what would change the conclusions. Ordered by value for the effort.

## Done (v0.1.0)

* Library: signals, selective-risk metrics, Learn-then-Test certification, hashed artifacts, the pure gate.
* Serving: llama-server and Ollama adapters with a logprob self-test, cascade router with full cost accounting,
  OpenAI-compatible server, privacy-by-default audit log, Prometheus metrics, drift alarm, shadow mode, Docker.
* Evaluation: pre-registered harness, two frozen experiments (TriviaQA, SQuAD 2.0) with 50 re-splits each,
  independent red-team, evaluation, governance and release audits (`docs/process/`).
* Findings: a 21-fold cut in wrong answers at a certified 10% error target, shift detection, quantization effects,
  a reproduced llama.cpp logprob bug, and a break-even model that explains the cost results.

## Next, in order

1. **Serve over a real socket with the server's own adapter.** Done in a first form: the harness-certified
   artifact was served through the deployed app on the T4s (`docs/RESULTS.md`, end-to-end demo). What remains is
   carrying the harness backends' stop-sequence and token-trimming handling into the server's own llama-server
   adapter, reconciling the build identity string it reports, and repeating the demo over a real TCP socket.
2. **A stronger or cheaper small tier, pre-registered.** The break-even model says what is needed: a small tier that
   resolves more than `cost_small / cost_large` of queries at the target error. Candidate directions: a small model
   fine-tuned for the task, a larger quality gap with a much cheaper small tier, or an easier traffic mix. The
   pre-registration comes first; any result is published.
3. **Per-request backend identity checks** (identity is currently re-checked periodically) and **signed artifacts**.
4. **Multi-target certification under the one-read rule for the calibration split.** Certifying several error targets
   from one pass over the split would complete the cost-versus-error frontier with guarantees at every point;
   today the frontier curves are empirical.
5. **A judge model for SimpleQA** (a third model, different from both tiers) so the shift hypothesis can be fully graded.
6. **Contamination probes, cross-template transfer and a quantization null control** (kill tests 4 and 11 in the
   design review).
7. **Upstream:** a minimal reproduction and fix for the llama.cpp logprob issue (31451), and the calibration metric
   request in lm-evaluation-harness (2189), submitted by the maintainer in line with each project's AI policy.

## What would change the conclusions

* If a small tier resolves more than its break-even share at a certified error level on some task, the cost claim
  flips for that task. Nothing measured so far says it cannot; nothing measured so far says it can.
* If a held-out, different-distribution evaluation shows the bound holding under shift, the "fragile under shift"
  finding needs revisiting. The current evidence (NQ-open, 50 of 50 re-splits violated) points the other way.
