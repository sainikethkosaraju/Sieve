# Sieve

Sieve is a confidence gate and OpenAI-compatible proxy for small-to-large LLM cascades on local hardware.
For each query it decides whether the small model's answer can be returned (**answer**), whether a larger model
should be asked (**escalate**), or whether neither is reliable enough (**abstain**). Thresholds come from a
calibration procedure (Learn-then-Test) that targets a stated bound on the error rate among answered queries,
and the gate fails closed when the model's logprobs or the calibration do not match what it was fitted on.

> **Disclosure.** This project was built with heavy AI assistance (Anthropic's Claude, via Claude Code).
> The git history carries `Co-Authored-By` trailers. The owner directed the work and reviews and is
> responsible for it; see [`docs/process/`](docs/process/) for how the AI-assisted process was audited.

## What was measured, and what was not

A pre-registered run on two Tesla T4s (Qwen2.5-1.5B small tier, Qwen2.5-7B large tier, llama.cpp, TriviaQA /
NQ-open / MMLU-Redux; details and caveats in **[docs/RESULTS.md](docs/RESULTS.md)**):

| Question | Result |
|---|---|
| Did the risk guarantee hold in distribution? | Test risk exceeded the 10% target in 5 of 50 re-splits (10%, 95% CI 3% to 22%): consistent with the guarantee, not proof. |
| Does it hold under shift? | No: on NQ-open, risk was 0.42 against a 0.10 target in 50 of 50 re-splits. A label-free drift alarm fired on all 10 windows. |
| Did the cascade save cost? | **No.** At 10% certified risk the 1.5B model's confidence let it answer about 4% of queries on average across the 50 re-splits (none on the canonical split), so the system behaved like the 7B model plus the cost of the wasted small-model attempt (about 0.90 s vs 0.68 s per correct answer). |
| Which signal is best? | Length-normalised and minimum token logprob beat max-softmax by about +0.03 AUROC; self-consistency (k=5) was worse and 1.7x slower than the large model. |
| Effect of quantization on the gate | Ranking quality dropped at 2-bit weights and at an 8-bit KV cache; a 4-bit KV cache produced non-terminating, nonsensical output from the 1.5B model in this setup (accuracy 0.0; possibly specific to this build and model). Thresholds fitted at f16 were violated in at most 8.2% of re-splits on every other configuration (target 10%). |

**Not established:** energy savings (power was throttled and confounded; no such claim is made), results for other
model families or sizes, behaviour on non-QA tasks, SimpleQA (no independent judge), a contamination check, or
anything measured on consumer GPUs other than Tesla T4. TriviaQA/NQ-open are likely in pretraining data. A negative
cost result for this particular model pair does not say cascades cannot work with a larger quality gap.

## What the guarantee says (and does not)

For exchangeable traffic, a frozen model/quantization/KV-cache type/prompt template/sampler and a fixed grader:
with probability at least `1 - delta` over the draw of the calibration set, the grader-judged error rate among
answered queries is at most `alpha`. It is not per-query, not conditional on topic or difficulty, not about factual
truth (only the grader's verdict, here strict exact match), not robust to distribution shift, and not valid after
changing any element of the configuration. The gate refuses mismatched artifacts. With small calibration sets the
procedure often certifies nothing; that is reported as an explicit empty feasible set.

## What is in the box

- `sieve.signals`, `sieve.calibrate`, `sieve.gate`: confidence signals, selective-prediction metrics, calibration with
  hashed artifacts, the pure decision function. Property- and simulation-tested.
- `sieve.backends`, `sieve.router`, `sieve.server`: llama-server and Ollama adapters, the cascade router with cost
  accounting that includes wasted attempts, and an OpenAI-compatible server (JSON and SSE) with backpressure,
  deadlines and cancellation. Tested against a mock backend and, for the llama-server parser and logprob self-test,
  against real responses captured on a T4. **The server has not been load-tested against a real GPU backend, and
  Docker images were not built in CI.**
- `sieve.observe`: privacy-by-default audit log, bounded-cardinality Prometheus metrics, drift alarm, shadow mode.
- `sieve.eval`: pre-registered evaluation harness (hash-locked splits, resumable cache, paired bootstrap, McNemar,
  TOST, hypothesis report), plus Kaggle drivers. Synthetic demo output lives in `results/synthetic-demo/` and is
  labelled synthetic everywhere.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint type test        # ruff, mypy, pytest: no GPU, no network
make reproduce-mock        # end-to-end synthetic pipeline, about a minute
```

Serving (mock backend, for a look at the API): see [docs/server.md](docs/server.md). Reproducing the real run:
[docs/kaggle.md](docs/kaggle.md) and [docs/reproduce.md](docs/reproduce.md).

## Layout

```
src/sieve/        library, server, observability, evaluation harness
benchmarks/       experiment configs (pre-registered) and the mock pipeline script
kaggle/           Kaggle GPU drivers (push, resume, decode results from the log)
results/          committed evidence: real aggregates (results/evidence) and a labelled synthetic demo
docs/RESULTS.md   measured results with limitations
docs/PREREGISTRATION.md   frozen protocol with dated deviations
docs/research/    background notes (AI-assisted, partly unverified)
docs/process/     how the project was planned, reviewed and audited
```

## Models and data

Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct are Apache-2.0. Datasets keep their own licences (see `NOTICE`);
item text and generations are not redistributed here. Note that Qwen2.5-3B is under a non-commercial research
licence and is not used.

## Known limitations of the software

See `docs/process/council/reviews/fixwave-notes.md` for the audit findings that were deliberately left open
(per-request backend identity checks, artifact signing, a global escalation guard, header-level slowloris
protection off loopback, several statistical-power items).

## License

MIT, see `LICENSE`. Cite via `CITATION.cff`.
