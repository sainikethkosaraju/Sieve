# Sieve Kaggle run: t4_squad

elapsed 4314 s; repo commit bd276d1fddd40365b969d4e76e88a34cbc1f807e; stages {'bootstrap': {'ok': True, 's': 13.8}, 'binary': {'ok': True, 's': 5.8}, 'models': {'ok': True, 's': 70.2}, 'probe': {'ok': True, 's': 33.0}, 'prepare': {'ok': True, 's': 12.0}, 'run': {'ok': True, 's': 4065.6}, 'analyse': {'ok': False, 's': 105.0}, 'report': {'ok': True, 's': 1.0}, 'sanity': {'ok': True, 's': 8.0}, 'emit': {'ok': True, 's': 0.0}}
- binary_source: data source: /tmp/llama-server-cand0
- llama_server_version: version: 0.8.0-dev (build 1, commit fac3a89)
built with GNU 13.3.0 for Linux x86_64
- versions: numpy 2.1.3 scipy 1.16.3 httpx 0.28.1 datasets 4.8.5 pandas 2.3.3
- run_rc: 0
- run_interrupted_by_deadline: False

## Problems
- analyse failed rc=1: st recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/tmp/sieve_repo/src/sieve/eval/__main__.py", line 3, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "/tmp/sieve_repo/src/sieve/eval/cli.py", line 119, in main
    rep = build_report(res, cfg)
  File "/tmp/sieve_repo
- report failed rc=1:     raise SystemExit(main())
                     ~~~~^^
  File "/tmp/sieve_repo/src/sieve/eval/cli.py", line 134, in main
    rep = build_report(res, cfg)
  File "/tmp/sieve_repo/src/sieve/eval/hypotheses.py", line 426, in build_report
    raise HypothesisReportRefusedError(
        f"coded criteria no longer match the pre-registration text: {crit_missing}"
    )
sieve.eval.hy

## Models (sha256 verified against the Hub)
| file | size | verified |
|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 1117320736 | True |
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4683074240 | True |

## Datasets manifest
```
{"cross_dataset_question_overlap": {}, "datasets": {"squad_v2": {"config": "squad_v2", "dataset_hash": "f3a1c31f0e1c8f0916121704bd69618c72ec2e81efb08c941b3e820a66178b67", "grader": "alias_em", "licence": "CC-BY-SA-4.0", "n": 6000, "pool_size": 11873, "repo": "rajpurkar/squad_v2", "revision": "3ffb306f725f7d2ce8394bc1873b24868140c412", "seed": 20261108, "source": "huggingface", "split": "validation"}}, "splits": {"index_hash": "d03e5c3d7d541d1df0077789562a6c0e302202c98f3166054b4394affca7d1e6", "n_splits": 50}, "synthetic": false}
```

## Per cell and dataset
| cell | dataset | n | acc | empty | defects | p50 wall s | tok/s p50 | first-tok p |
|---|---|---|---|---|---|---|---|---|
| L_q4km | squad_v2 | 6000 | 0.7107 | 0 | {'missing_logprobs': 2, 'truncated': 17} | 0.4339 | 7.68 | 0.9082 |
| S_q4km | squad_v2 | 6000 | 0.5417 | 0 | {'missing_logprobs': 6, 'truncated': 174} | 0.1212 | 28.64 | 0.6604 |

## Servers
```
{"L_q4km": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "1", "log_flash_attn": null, "model_hash": "65b8fcd92af6b4fe", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": false}, "server_start_s": 2.867}, "S_q4km": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "0", "log_flash_attn": null, "model_hash": "6a1a2eb6d15622bf", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": true}, "server_start_s": 2.169}}
```

## Energy / VRAM (nvidia-smi windows)
```
{"block": 0, "cell": "S_q4km", "complete": true, "duration_s": 277.960416825, "idle_watts": 4.734877049180327, "joules_gross": 12985.085589049642, "joules_net": 10352.868792639063, "mean_watts_net": 37.24583849346105, "n_generated": 1550, "n_samples": 4224, "peak_vram_mib": 4665.0, "state": "measured", "throttled": true}
{"block": 0, "cell": "L_q4km", "complete": true, "duration_s": 745.6960530790001, "idle_watts": 4.734877049180327, "joules_gross": 35011.1091410785, "joules_net": 27949.550886302277, "mean_watts_net": 37.48115706244896, "n_generated": 1500, "n_samples": 11346, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "S_q4km", "complete": true, "duration_s": 262.21706096699995, "idle_watts": 4.734877049180327, "joules_gross": 12444.811002548473, "joules_net": 9961.67991479614, "mean_watts_net": 37.990205054009884, "n_generated": 1550, "n_samples": 3980, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "L_q4km", "complete": true, "duration_s": 736.6093179790003, "idle_watts": 4.734877049180327, "joules_gross": 34610.62835512076, "joules_net": 27635.119247298477, "mean_watts_net": 37.516657165184434, "n_generated": 1500, "n_samples": 11214, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 2, "cell": "S_q4km", "complete": true, "duration_s": 256.74954204000005, "idle_watts": 4.734877049180327, "joules_gross": 12249.495830959942, "joules_net": 9818.14080297443, "mean_watts_net": 38.24014923245636, "n_generated": 1550, "n_samples": 3894, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 2, "cell": "L_q4km", "complete": true, "duration_s": 740.6520646900003, "idle_watts": 4.734877049180327, "joules_gross": 34763.76329882131, "joules_net": 27749.970373763896, "mean_watts_net": 37.46694527258037, "n_generated": 1500, "n_samples": 11274, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 3, "cell": "S_q4km", "complete": true, "duration_s": 267.28417062100016, "idle_watts": 4.734877049180327, "joules_gross": 12737.96958817325, "joules_net": 10206.854218008106, "mean_watts_net": 38.187275341797466, "n_generated": 1550, "n_samples": 4050, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
{"block": 3, "cell": "L_q4km", "complete": true, "duration_s": 736.3301696650001, "idle_watts": 4.734877049180327, "joules_gross": 34595.76101295239, "joules_net": 27622.89537102066, "mean_watts_net": 37.5142789322186, "n_generated": 1500, "n_samples": 11214, "peak_vram_mib": 4697.0, "state": "measured", "throttled": true}
```

## Output samples

Removed: sample rows contained dataset text, which is not redistributed (see PROVENANCE.md and NOTICE).
