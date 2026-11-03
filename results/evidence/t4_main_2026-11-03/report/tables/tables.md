### accuracy

| cell | tier | quant | kv | dataset | n | n_graded | accuracy | fail_closed_rate | mean_time_s | wasted_time_share |
|---|---|---|---|---|---|---|---|---|---|---|
| L_q4km | L | Q4_K_M | f16 | mmlu_redux | 1500 | 1500 | 0.743 | 0.000 | 0.234 | 0.000 |
| L_q4km | L | Q4_K_M | f16 | nq_open | 2000 | 2000 | 0.196 | 0.002 | 0.240 | 0.000 |
| L_q4km | L | Q4_K_M | f16 | simpleqa_verified | 1000 | 0 | n/a | 0.001 | 0.280 | 0.000 |
| L_q4km | L | Q4_K_M | f16 | triviaqa | 6000 | 6000 | 0.559 | 0.000 | 0.196 | 0.000 |
| L_q4km_kvq8 | L | Q4_K_M | q8_0 | mmlu_redux | 1500 | 1500 | 0.749 | 0.000 | 0.249 | 0.000 |
| L_q4km_kvq8 | L | Q4_K_M | q8_0 | nq_open | 2000 | 2000 | 0.193 | 0.002 | 0.240 | 0.000 |
| L_q4km_kvq8 | L | Q4_K_M | q8_0 | simpleqa_verified | 1000 | 0 | n/a | 0.002 | 0.281 | 0.000 |
| L_q4km_kvq8 | L | Q4_K_M | q8_0 | triviaqa | 6000 | 6000 | 0.562 | 0.000 | 0.197 | 0.000 |
| S_f16 | S | f16 | f16 | triviaqa | 6000 | 6000 | 0.384 | 0.001 | 0.089 | 0.000 |
| S_q2k | S | Q2_K | f16 | triviaqa | 6000 | 6000 | 0.149 | 0.140 | 0.107 | 0.000 |
| S_q3km | S | Q3_K_M | f16 | triviaqa | 6000 | 6000 | 0.318 | 0.001 | 0.063 | 0.000 |
| S_q4km | S | Q4_K_M | f16 | mmlu_redux | 1500 | 1500 | 0.595 | 0.000 | 0.066 | 0.000 |
| S_q4km | S | Q4_K_M | f16 | nq_open | 2000 | 2000 | 0.119 | 0.015 | 0.075 | 0.000 |
| S_q4km | S | Q4_K_M | f16 | simpleqa_verified | 1000 | 0 | n/a | 0.008 | 0.089 | 0.000 |
| S_q4km | S | Q4_K_M | f16 | triviaqa | 6000 | 6000 | 0.366 | 0.001 | 0.063 | 0.000 |
| S_q4km_kvq4 | S | Q4_K_M | q4_0 | triviaqa | 6000 | 6000 | 0.000 | 0.974 | 0.341 | 0.000 |
| S_q4km_kvq8 | S | Q4_K_M | q8_0 | triviaqa | 6000 | 6000 | 0.366 | 0.001 | 0.064 | 0.000 |
| S_q8 | S | Q8_0 | f16 | triviaqa | 6000 | 6000 | 0.383 | 0.001 | 0.073 | 0.000 |

### methods

| method | certified_splits | empty_splits | violation_freq | mean_test_risk | mean_coverage | mean_cost_per_correct | split0_risk | split0_coverage | split0_escalation | split0_cost_per_correct |
|---|---|---|---|---|---|---|---|---|---|---|
| L_only | 50 | 0 | 0.040 | 0.075 | 0.273 | 0.792 | 0.068 | 0.307 | 1.000 | 0.683 |
| S_only | 44 | 6 | 0.000 | 0.045 | 0.050 | 1.323 | 0.029 | 0.051 | 0.949 | 1.249 |
| gate:entropy | 49 | 1 | 0.163 | 0.076 | 0.257 | 1.296 | 0.074 | 0.306 | 0.950 | 0.881 |
| gate:margin | 6 | 44 | 0.833 | 0.112 | 0.055 | 5.323 | n/a | n/a | n/a | n/a |
| gate:max_softmax | 0 | 50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| gate:mean_logprob | 50 | 0 | 0.100 | 0.074 | 0.258 | 1.335 | 0.068 | 0.307 | 1.000 | 0.901 |
| gate:primary | 50 | 0 | 0.100 | 0.074 | 0.258 | 1.335 | 0.068 | 0.307 | 1.000 | 0.901 |
| query_router | 44 | 6 | 0.000 | 0.025 | 0.073 | 3.429 | n/a | n/a | n/a | n/a |
| random_router | 50 | 0 | 0.040 | 0.075 | 0.273 | 0.792 | 0.068 | 0.307 | 1.000 | 0.683 |
| self_consistency | 17 | 33 | 0.118 | 0.059 | 0.221 | 3.763 | n/a | n/a | n/a | n/a |

### signal_auroc

| tier | signal | auroc | lo | hi |
|---|---|---|---|---|
| L | first_margin | 0.816 | 0.797 | 0.835 |
| L | max_prob | 0.845 | 0.828 | 0.862 |
| L | mean_logprob | 0.866 | 0.850 | 0.881 |
| L | mean_margin | 0.784 | 0.763 | 0.804 |
| L | min_logprob | 0.869 | 0.853 | 0.884 |
| L | neg_first_entropy | 0.851 | 0.833 | 0.867 |
| L | neg_mean_entropy | 0.867 | 0.851 | 0.882 |
| S | first_margin | 0.740 | 0.717 | 0.763 |
| S | max_prob | 0.808 | 0.789 | 0.827 |
| S | mean_logprob | 0.837 | 0.819 | 0.855 |
| S | mean_margin | 0.744 | 0.721 | 0.766 |
| S | min_logprob | 0.843 | 0.825 | 0.860 |
| S | neg_first_entropy | 0.811 | 0.791 | 0.829 |
| S | neg_mean_entropy | 0.825 | 0.806 | 0.844 |

### comparisons_gate_minus_other

| comparison | acc_on_answered_diff | cost_per_correct_diff | coverage_diff | mcnemar_p |
|---|---|---|---|---|
| vs_L_only | 0.000 [0.000, 0.000] | 0.218 [0.203, 0.235] | 0.000 [0.000, 0.000] | 1.000 |
| vs_S_only | -0.039 [-0.074, 0.001] | -0.348 [-0.618, -0.156] | 0.256 [0.237, 0.276] | 0.000 |
| vs_gate:entropy | 0.005 [-0.001, 0.012] | 0.020 [0.005, 0.034] | 0.002 [-0.003, 0.006] | 0.238 |
| vs_gate:mean_logprob | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| vs_random_router_certified | 0.000 [0.000, 0.000] | 0.218 [0.203, 0.234] | 0.000 [0.000, 0.000] | 1.000 |
| vs_query_router@matched | 0.000 [0.000, 0.000] | 0.218 [0.203, 0.235] | 0.000 [0.000, 0.000] | 1.000 |
| vs_random_router@matched | 0.000 [0.000, 0.000] | 0.218 [0.203, 0.235] | 0.000 [0.000, 0.000] | 1.000 |

### shift

| dataset | graded | n | violation_freq | mean_risk | risk_ci95_split0 | drift_alarm_rate | n_windows |
|---|---|---|---|---|---|---|---|
| nq_open | yes | 2000 | 1.000 | 0.422 | 0.451 [0.390, 0.512] | 1.000 | 10 |
| simpleqa_verified | no | 1000 | n/a | n/a | n/a | 1.000 | 5 |

### quantization

| config | transferred_violation_freq | recalibrated_violation_freq | delta_auroc | delta_ece | auroc_equivalent | ece_equivalent |
|---|---|---|---|---|---|---|
| kv_q4 | 0.041 | 0.120 | n/a | n/a | n/a | n/a |
| kv_q8 | 0.082 | 0.080 | -0.020 [-0.035, -0.006] | 0.002 [-0.015, 0.020] | no | yes |
| ladder_q2k | 0.041 | 0.040 | -0.043 [-0.072, -0.014] | -0.025 [-0.046, -0.009] | no | no |
| ladder_q3km | 0.082 | 0.062 | -0.005 [-0.023, 0.014] | 0.003 [-0.017, 0.021] | no | yes |
| ladder_q4km | 0.082 | 0.100 | -0.014 [-0.029, 0.000] | -0.000 [-0.019, 0.017] | no | yes |
| ladder_q8 | 0.061 | 0.041 | -0.004 [-0.009, 0.001] | 0.001 [-0.012, 0.011] | yes | yes |

### hypotheses

| id | title | verdict | why |
|---|---|---|---|
| H1 | validity | INCONCLUSIVE | SUPPORTED-WEAK: violation frequency 0.100 <= 15% but the Clopper-Pearson 95% upper bound 0.218 exceeds it, so the re-splits cannot exclude a higher violation rate (reported as INCONCLUSIVE) |
| H2 | signal | SUPPORTED | a candidate signal clears +0.02 with CI above 0 |
| H3 | cascade value | NOT SUPPORTED | not distinguishable from (or worse than) the matched random router; no cost-per-correct advantage over L-only once wasted S attempts are charged |
| H4 | self-consistency | NOT SUPPORTED | no gain of >= 0.03 AUROC |
| H5 | shift | INCONCLUSIVE | SimpleQA Verified risk unavailable (judge: simpleqa-judge:unavailable) |
| H6 | quantization | INCONCLUSIVE | neither the shift nor the equivalence criterion is met for all configs |
| H7 | hard sets | NOT SUPPORTED | AUROC 0.792 |
