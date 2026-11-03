# Reproducing the evaluation

## 1. Offline mock (no GPU, no network)

```sh
.venv/bin/python -m pip install -e ".[dev,eval]"      # eval extra: datasets, matplotlib, pandas; also install pyyaml
benchmarks/reproduce_mock.sh /tmp/sieve-mock results/synthetic-demo   # ~20 s
# equivalent: python -m sieve.eval mock --workdir /tmp/sieve-mock --out results/synthetic-demo
```

This runs prepare-data, the cache-everything runner (fake backend, injected failures, ABAB
blocks, synthetic power), the audit sheet, certification on 50 re-splits, all baselines, the
hypothesis report and the HTML report. **Everything it produces is synthetic**: the page,
figure footers, hypothesis text and `results.json` say so. It tests the pipeline, not a model.
The committed copy is `results/synthetic-demo/index.html` (the folder was renamed from
`results/mock`; the default OUT in `benchmarks/reproduce_mock.sh` still says `results/mock`, so pass
the second argument). Every artefact carries an explicit SYNTHETIC marker (a `synthetic` column in
each CSV, a header in `tables.md`, banners in html/json, PNG metadata plus footer) and model/dataset
labels read `synthetic-...`; `tests/eval/test_synthetic_marker.py` fails if one does not.

To expose it as `make reproduce-mock`, replace the Phase A recipe in the `Makefile` with
`benchmarks/reproduce_mock.sh` (the Makefile is not owned by this seat); the Phase A simulation
test remains available as `pytest tests/test_ltt_simulation.py`.

Quality gates (all expected to pass): `ruff check src tests`, `ruff format --check src tests`,
`mypy`, `pytest` (the `slow` marker covers the pipeline tests, still run by default).

## 2. Real run on Kaggle 2x T4

Prerequisites: llama.cpp built at a recorded commit; GGUF files from
`Qwen/Qwen2.5-3B-Instruct-GGUF` (fp16, q8_0, q4_k_m, q3_k_m, q2_k) and
`Qwen/Qwen2.5-7B-Instruct-GGUF` (q4_k_m), all verified to exist on 2026-11-03; the 3B repo's
licence tag is "other" (Qwen research licence), the 7B's is Apache-2.0. If the 3B ladder is
unavailable use `benchmarks/t4_fallback_1p5b.yaml` and record a dated Deviation first.

1. Edit `benchmarks/t4_main.yaml`: set `backend.llama_cpp_commit`; set `judge` (see below).
   `preregistration.frozen_sha256` is already pinned; it must match
   `python -c "from sieve.eval import prereg; print(prereg.frozen_hash('docs/PREREGISTRATION.md'))"`.
2. `python -m sieve.eval prepare-data --config benchmarks/t4_main.yaml --workdir $W` and read
   the printed counts, revisions, licences and cross-dataset overlap. Do this once; the split
   index is frozen and re-running with different inputs is refused.
3. `python -m sieve.eval run --config benchmarks/t4_main.yaml --workdir $W` (add `--cell NAME`
   to run a subset, `--no-power` to skip nvidia-smi). Kaggle sessions die: just run the same
   command again; finished generations are skipped, a torn last line is dropped. Keep `$W` on the
   persistent output dataset between sessions.
4. `python -m sieve.eval audit-sheet ...`, label `audit/audit_sheet.csv` by hand
   (`human_label` = `correct` or `wrong`), then `python -m sieve.eval audit-analyse --workdir $W`.
5. `python -m sieve.eval analyse --config ... --workdir $W` (reads split B once per re-split and
   stores certifications; re-running it reuses them), then `report --out results/<run-name>`.

`benchmarks/t4_smoke.yaml` is a tiny configuration for validating the pipeline on Kaggle; its
output must not be reported as the pre-registered experiment.

### Backend provider contract

`backend.provider: "package.module:factory"`; `factory(cfg: ExperimentConfig)` returns an object
with `open(cell: CellSpec) -> ContextManager[Backend]`. Loading the model / launching the server
happens in `open` (its duration is stored as `load_s`). The `Backend` is the Phase A Protocol
(`info`, `async generate(prompt, params, *, n_probs) -> Completion`) and must: send the prompt
raw to the completion endpoint with no chat template; set `info.template_hash` to
`sieve.eval.prompts.template_hash()`; return per-token logprobs with top-k lists and `n_gen`;
report `finish` honestly (`error` makes the attempt count as wasted); and fail `open` if its
logprob self-test fails. `src/sieve/eval/_fake_backend.py` is the reference implementation.

### SimpleQA judge

`judge.kind: unavailable` (default) leaves SimpleQA ungraded and H5 INCONCLUSIVE. To grade it,
set `judge.kind: backend` and `judge.factory: "module:fn"` returning a `BackendJudge` over a model
that differs from S and L (document the model and its prompt in the Deviations section).

### Where things land

`$W/analysis/results.json` (schema: `results/schema/results.schema.json`),
`$W/analysis/hypotheses.txt`, `$W/report/index.html` and `tables/*.csv`. Commit only small
artefacts under `results/<run-name>/` (report page, tables, figures, `results.json`), never the
generation cache.

## 3. Determinism

With the same config the mock reproduces `results.json`, every table and the generation cache
byte for byte (simulated clock, hash-seeded randomness; `meta.commit` aside). Real runs are
deterministic given the cache; the generation itself is as deterministic as the backend (the
repeat-run subset measures this).
