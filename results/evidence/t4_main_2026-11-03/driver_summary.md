# Sieve Kaggle run: t4_main

elapsed 552 s; repo commit bb7fed71e744c7bb6316816074bda12c96ea691c+dirty; stages {'bootstrap': {'ok': True, 's': 33.9}, 'binary': {'ok': True, 's': 6.7}, 'models': {'ok': True, 's': 130.9}, 'probe': {'ok': True, 's': 22.0}, 'prepare': {'ok': True, 's': 0.0}, 'run': {'ok': True, 's': 88.0}, 'analyse': {'ok': True, 's': 234.0}, 'report': {'ok': True, 's': 3.0}, 'sanity': {'ok': True, 's': 34.0}, 'emit': {'ok': True, 's': 0.0}}
- binary_source: data source: /tmp/llama-server-cand0
- llama_server_version: version: 0.8.0-dev (build 1, commit fac3a89)
built with GNU 13.3.0 for Linux x86_64
- versions: numpy 2.1.3 scipy 1.16.3 httpx 0.28.1 datasets 4.8.5 pandas 2.3.3
- resumed_from: /kaggle/input/notebooks/ksainiketh/sieve-t4-main-1/sieve_work_t4_main
- run_rc: 0
- run_interrupted_by_deadline: False

## Models (sha256 verified against the Hub)
| file | size | verified |
|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 1117320736 | True |
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4683074240 | True |
| qwen2.5-1.5b-instruct-fp16.gguf | 3560416288 | True |
| qwen2.5-1.5b-instruct-q8_0.gguf | 1894532128 | True |
| qwen2.5-1.5b-instruct-q3_k_m.gguf | 924455968 | True |
| qwen2.5-1.5b-instruct-q2_k.gguf | 752880160 | True |

## Datasets manifest
```
{"cross_dataset_question_overlap": {"mmlu_redux": 0, "nq_open": 1, "simpleqa_verified": 0}, "datasets": {"mmlu_redux": {"config": null, "dataset_hash": "45f071f6ba6863235a700a9754ae0bb3d10a2f2c5d6c8bb5599e8a65956a3825", "grader": "mmlu_letter", "licence": "CC-BY-4.0", "n": 1500, "pool_size": 5330, "repo": "edinburgh-dawg/mmlu-redux-2.0", "revision": "372ea425445d51e1ba1188c56e5e893f8138621f", "seed": 20261106, "source": "huggingface", "split": "test"}, "nq_open": {"config": "nq_open", "dataset_hash": "c22e0e8d7aa9cb592035b05b80149f30034971262f631bf7b255d9ca2037e591", "grader": "alias_em", "licence": "CC-BY-SA-3.0", "n": 2000, "pool_size": 3610, "repo": "google-research-datasets/nq_open", "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28", "seed": 20261104, "source": "huggingface", "split": "validation"}, "simpleqa_verified": {"config": null, "dataset_hash": "df3eee9218af84a91dc48edca294355f7df926a9c7dd9feb9856303e665df200", "grader": "simpleqa_judge", "licence": "MIT", "n": 1000, "pool_size": 1000, "repo": "google/simpleqa-verified", "revision": "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc", "seed": 20261105, "source": "huggingface", "split": "eval"}, "triviaqa": {"config": "rc.nocontext", "dataset_hash": "cc8a94f8a7d9f4d3e1a0681ba4536c9080c9893dde3a54d5cacb2e3ad03b8ee9", "grader": "alias_em", "licence": "unknown (dataset card lists no licence; check upstream terms before redistribution)", "n": 6000, "pool_size": 17944, "repo": "mandarjoshi/trivia_qa", "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "seed": 20261103, "source": "huggingface", "split": "validation"}}, "splits": {"index_hash": "3c682f29233e383b6fee63662bc649cb131eccdfe26fab0cfa4c54045e871b36", "n_splits": 50}, "synthetic": false}
```

## Per cell and dataset
| cell | dataset | n | acc | empty | defects | p50 wall s | tok/s p50 | first-tok p |
|---|---|---|---|---|---|---|---|---|
| L_q4km | mmlu_redux | 1500 | 0.7433 | 0 | {} | 0.2121 | 4.72 | 0.9283 |
| L_q4km | nq_open | 2000 | 0.196 | 0 | {'truncated': 3} | 0.21 | 15.25 | 0.6596 |
| L_q4km | simpleqa_verified | 1000 | None | 0 | {'truncated': 1} | 0.2638 | 16.88 | 0.5887 |
| L_q4km | triviaqa | 6000 | 0.5593 | 0 | {} | 0.1811 | 11.87 | 0.7565 |
| L_q4km_kvq8 | mmlu_redux | 1500 | 0.7493 | 0 | {} | 0.2213 | 4.52 | 0.9302 |
| L_q4km_kvq8 | nq_open | 2000 | 0.193 | 0 | {'missing_logprobs': 1, 'truncated': 2} | 0.2092 | 15.08 | 0.6584 |
| L_q4km_kvq8 | simpleqa_verified | 1000 | None | 0 | {'truncated': 2} | 0.2662 | 16.95 | 0.5907 |
| L_q4km_kvq8 | triviaqa | 6000 | 0.5617 | 0 | {'truncated': 1} | 0.1824 | 11.78 | 0.7573 |
| S_f16 | triviaqa | 6000 | 0.3843 | 0 | {'truncated': 5} | 0.077 | 26.93 | 0.5474 |
| S_q2k | triviaqa | 6000 | 0.1495 | 0 | {'missing_logprobs': 1, 'truncated': 839} | 0.0679 | 43.65 | 0.2893 |
| S_q3km | triviaqa | 6000 | 0.3183 | 0 | {'missing_logprobs': 2, 'truncated': 4} | 0.0589 | 35.93 | 0.5106 |
| S_q4km | mmlu_redux | 1500 | 0.5953 | 0 | {} | 0.0594 | 16.92 | 0.7997 |
| S_q4km | nq_open | 2000 | 0.1195 | 0 | {'missing_logprobs': 2, 'truncated': 29, 'utf8_gap': 1} | 0.0642 | 47.42 | 0.4664 |
| S_q4km | simpleqa_verified | 1000 | None | 0 | {'truncated': 8} | 0.0837 | 55.55 | 0.4864 |
| S_q4km | triviaqa | 6000 | 0.3658 | 0 | {'missing_logprobs': 1, 'truncated': 6} | 0.0575 | 36.65 | 0.5256 |
| S_q4km_kvq4 | triviaqa | 6000 | 0.0 | 3 | {'empty_answer': 1, 'missing_logprobs': 2, 'truncated': 5838} | 0.3455 | 92.56 | 0.2455 |
| S_q4km_kvq8 | triviaqa | 6000 | 0.3665 | 0 | {'missing_logprobs': 1, 'truncated': 6} | 0.0593 | 35.41 | 0.5234 |
| S_q8 | triviaqa | 6000 | 0.383 | 0 | {'truncated': 5} | 0.0659 | 32.16 | 0.5471 |

## Servers
```
{"L_q4km": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "1", "log_flash_attn": null, "model_hash": "65b8fcd92af6b4fe", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": false}, "server_start_s": 3.292}, "L_q4km_kvq8": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=on|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "1", "log_flash_attn": null, "model_hash": "65b8fcd92af6b4fe", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": false}, "server_start_s": 2.226}, "S_f16": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "0", "log_flash_attn": null, "model_hash": "fc89e330deb3fd8f", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": true}, "server_start_s": 2.724}, "S_q2k": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "0", "log_flash_attn": null, "model_hash": "5ede348e91ce1e7a", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": true}, "server_start_s": 1.63}, "S_q3km": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "0", "log_flash_attn": null, "model_hash": "58cb5c05ecef48e8", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": true}, "server_start_s": 1.639}, "S_q4km": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f|b1-fac3a89|fa=auto|ngl=99|stop=\"\\n\"", "build_info": "b1-fac3a89", "gpu": "0", "log_flash_attn": null, "model_hash": "6a1a2eb6d15622bf", "self_test": {"count_matches_tokens": true, "finite_nonpositive": true, "greedy_deterministic": true, "top_k_contains_chosen": true, "utf8_reassembles": true}, "server_start_s": 1.614}, "S_q4km_kvq4": {"backend_build": "llama.cpp@fac3a894fc04901c3e5abaf
```

## Energy / VRAM (nvidia-smi windows)
```
{"block": 0, "cell": "S_f16", "complete": true, "duration_s": 130.18486686099993, "idle_watts": 4.798060109289617, "joules_gross": 5017.936429448919, "joules_net": 3768.6667964110316, "mean_watts_net": 28.9485781817858, "n_generated": 1500, "n_samples": 1830, "peak_vram_mib": 5153.0, "state": "measured", "throttled": true}
{"block": 0, "cell": "S_q8", "complete": true, "duration_s": 108.80209346000004, "idle_watts": 4.798060109289617, "joules_gross": 4179.81719602324, "joules_net": 3135.7392271479866, "mean_watts_net": 28.820578055336856, "n_generated": 1500, "n_samples": 1536, "peak_vram_mib": 5165.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "S_f16", "complete": true, "duration_s": 133.680652109, "idle_watts": 4.798060109289617, "joules_gross": 5098.664601505853, "joules_net": 3815.8489929698217, "mean_watts_net": 28.544512109788858, "n_generated": 1500, "n_samples": 1886, "peak_vram_mib": 5171.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "S_q8", "complete": true, "duration_s": 108.7871390790001, "idle_watts": 4.798060109289617, "joules_gross": 4156.528235897058, "joules_net": 3112.593771059674, "mean_watts_net": 28.611780743671737, "n_generated": 1500, "n_samples": 1536, "peak_vram_mib": 5175.0, "state": "measured", "throttled": true}
{"block": 2, "cell": "S_f16", "complete": true, "duration_s": 136.37091308699996, "idle_watts": 4.798060109289617, "joules_gross": 5160.104855995827, "joules_net": 3851.4731796955557, "mean_watts_net": 28.24262954988391, "n_generated": 1500, "n_samples": 1923, "peak_vram_mib": 5175.0, "state": "measured", "throttled": true}
{"block": 2, "cell": "S_q8", "complete": true, "duration_s": 111.12153146800006, "idle_watts": 4.798060109289617, "joules_gross": 4269.026404494647, "joules_net": 3202.6908296550832, "mean_watts_net": 28.821514492691904, "n_generated": 1500, "n_samples": 1570, "peak_vram_mib": 5175.0, "state": "measured", "throttled": true}
{"block": 3, "cell": "S_f16", "complete": true, "duration_s": 134.57779422, "idle_watts": 4.798060109289617, "joules_gross": 5114.035469315755, "joules_net": 3822.6107772294176, "mean_watts_net": 28.40446894961315, "n_generated": 1500, "n_samples": 1899, "peak_vram_mib": 5175.0, "state": "measured", "throttled": true}
{"block": 3, "cell": "S_q8", "complete": true, "duration_s": 109.74812952899993, "idle_watts": 4.798060109289617, "joules_gross": 4199.253002220217, "joules_net": 3146.0967574957285, "mean_watts_net": 28.66651824498204, "n_generated": 1500, "n_samples": 1549, "peak_vram_mib": 5175.0, "state": "measured", "throttled": true}
{"block": 0, "cell": "S_q3km", "complete": true, "duration_s": 95.2257405119999, "idle_watts": 4.798060109289617, "joules_gross": 3655.3303541068403, "joules_net": 2741.532700250458, "mean_watts_net": 28.78982810225543, "n_generated": 1500, "n_samples": 1348, "peak_vram_mib": 2051.0, "state": "measured", "throttled": true}
{"block": 0, "cell": "S_q2k", "complete": true, "duration_s": 161.773359626, "idle_watts": 4.798060109289617, "joules_gross": 6214.495300914845, "joules_net": 4662.098693780297, "mean_watts_net": 28.81870478896212, "n_generated": 1500, "n_samples": 2290, "peak_vram_mib": 2063.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "S_q3km", "complete": true, "duration_s": 94.73028911400002, "idle_watts": 4.798060109289617, "joules_gross": 3621.4541978706943, "joules_net": 2712.4109551919823, "mean_watts_net": 28.632985083871343, "n_generated": 1500, "n_samples": 1342, "peak_vram_mib": 2067.0, "state": "measured", "throttled": true}
{"block": 1, "cell": "S_q2k", "complete": true, "duration_s": 156.23235192800007, "idle_watts": 4.798060109289617, "joules_gross": 5977.302311539454, "joules_net": 4478.077880406987, "mean_watts_net": 28.662935846160188, "n_generated": 1500, "n_samples": 2214, "peak_vram_mib": 2071.0, "state": "measured", "throttled": true}
```

## Output samples

Removed: sample rows contained dataset question text, gold answers and raw generations, which are not redistributed (see `PROVENANCE.md` and `NOTICE`).
