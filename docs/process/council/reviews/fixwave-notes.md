# Fix wave notes (after the red-team code audit and the eval audit)

Fixed: F-01 (shared `sieve.prompting`, D-019 calibrated template, answer span in the router), F-02, F-03, F-04, F-05 (as a documented tripwire), F-06, F-07, F-08 (H1 CI rule, H2 multiplicity NOT fixed), F-10 (limit_concurrency + corrected doc), F-13, F-14, F-17, F-18, F-21, eval-audit C1 (sanity gate only), H3 (NaN, matched coverage), H4/M8 (synthetic markers; Pareto coverage axis NOT added), M1 (audit rule on CI only), SimpleQA `not_attempted` in accuracy_on_answered.

## KNOWN LIMITATIONS (not fixed)
- F-08 (part): H2 still takes the best of three candidate signals with no Holm correction; re-splits remain dependent resamples (documented only).
- F-09: backend identity is polled (`/props` on `health_interval_s`), not verified per request; basename-only model file check.
- F-11: sampling-mismatch check is still vacuous (echo of requested params); client `max_tokens` is outside the identity.
- F-12: whole-generation / first-token confidence can be prompt-induced; needs a real-model injection test.
- F-15: escalation guard is global and can be used to shed legitimate hard requests (availability only).
- F-16: logprob determinism is only checked at startup, sequentially, not under concurrent batching.
- F-19: artifact hash is integrity, not authenticity; the validator does not check internal consistency (`risk_upper_bound <= alpha`).
- F-20: cache record hashes are self-referential (no chain/MAC).
- F-05 residue: B-once / C-hash remains bypassable (see docs/evaluation.md); ledger is not hash-chained.
- M2: H5 operationalisation (no per-window ordering, no false-alarm control on iid windows, few windows).
- M3: H4 is underpowered (SC evaluated only on the C0 slice of the 1000-item subset).
- M4: H3 query router is weak and has no abstention; coverage-at-certified-risk comparison not added.
- M5: H1 vs H6 threshold asymmetry; H2-H7 rest on one canonical re-split.
- M6: no per-stratum risk for TriviaQA (no strata assigned).
- M7: cost/energy accounting issues (unused L KV arm, co-resident VRAM, single idle baseline, throttle mask).
- M9: contamination probe and few-shot exclusion from eval items not done.
- Eval-audit C1 (rest): `stop` is still not part of `SamplingParams`/`cell_id`; the 50-item real-server canary has not been run (only the gate exists).
- Eval-audit H1/H2/H3: Deviations block, real GPU/driver/commit fields in `hardware_info()`, anchoring the prereg pin to the freeze commit and comparing cfg alpha/delta/n_splits/etc. to the prereg numbers: not done.
- Kill tests 4 (cross-template/cross-dataset), 10 (energy validity), 11 (quant null control): not implemented.
- `kaggle/run_experiment.py` embeds the repo sources and is now stale (test `test_committed_script_embeds_current_sources` fails); regenerate with `python kaggle/build_experiment_script.py` (kaggle/ was off limits for this wave). `benchmarks/reproduce_mock.sh` still defaults OUT to `results/mock`; pass `results/synthetic-demo`.  [Update: fixed; the script was rebuilt and the full suite passes.]
