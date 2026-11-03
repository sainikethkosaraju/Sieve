<!-- SYNTHETIC -->
**SYNTHETIC DATA (fake backend): nothing here is a result about a real model.**

### accuracy

| synthetic | cell | tier | quant | kv | dataset | n | n_graded | accuracy | fail_closed_rate | mean_time_s | wasted_time_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SYNTHETIC | synthetic-L_q4km | L | synthetic-Q4_K_M | f16 | synthetic-mmlu_redux | 300 | 300 | 0.733 | 0.000 | 0.855 | 0.008 |
| SYNTHETIC | synthetic-L_q4km | L | synthetic-Q4_K_M | f16 | synthetic-nq_open | 300 | 300 | 0.823 | 0.000 | 0.966 | 0.004 |
| SYNTHETIC | synthetic-L_q4km | L | synthetic-Q4_K_M | f16 | synthetic-simpleqa_verified | 200 | 200 | 0.505 | 0.000 | 0.949 | 0.002 |
| SYNTHETIC | synthetic-L_q4km | L | synthetic-Q4_K_M | f16 | synthetic-triviaqa | 1200 | 1200 | 0.902 | 0.000 | 0.968 | 0.004 |
| SYNTHETIC | synthetic-S_f16 | S | synthetic-f16 | f16 | synthetic-triviaqa | 1200 | 1200 | 0.718 | 0.000 | 0.156 | 0.005 |
| SYNTHETIC | synthetic-S_q3km | S | synthetic-Q3_K_M | f16 | synthetic-triviaqa | 1200 | 1200 | 0.642 | 0.000 | 0.219 | 0.003 |
| SYNTHETIC | synthetic-S_q4km | S | synthetic-Q4_K_M | f16 | synthetic-mmlu_redux | 300 | 300 | 0.470 | 0.000 | 0.171 | 0.003 |
| SYNTHETIC | synthetic-S_q4km | S | synthetic-Q4_K_M | f16 | synthetic-nq_open | 300 | 300 | 0.607 | 0.000 | 0.209 | 0.009 |
| SYNTHETIC | synthetic-S_q4km | S | synthetic-Q4_K_M | f16 | synthetic-simpleqa_verified | 200 | 200 | 0.210 | 0.000 | 0.202 | 0.007 |
| SYNTHETIC | synthetic-S_q4km | S | synthetic-Q4_K_M | f16 | synthetic-triviaqa | 1200 | 1200 | 0.699 | 0.000 | 0.213 | 0.005 |
| SYNTHETIC | synthetic-S_q4km_kvq4 | S | synthetic-Q4_K_M | q4_0 | synthetic-triviaqa | 1200 | 1200 | 0.694 | 0.000 | 0.212 | 0.005 |
| SYNTHETIC | synthetic-S_q8 | S | synthetic-Q8_0 | f16 | synthetic-triviaqa | 1200 | 1200 | 0.714 | 0.000 | 0.188 | 0.005 |

### methods

| synthetic | method | certified_splits | empty_splits | violation_freq | mean_test_risk | mean_coverage | mean_cost_per_correct | split0_risk | split0_coverage | split0_escalation | split0_cost_per_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SYNTHETIC | S_only | 43 | 7 | 0.047 | 0.063 | 0.531 | 0.434 | 0.092 | 0.573 | 0.427 | 0.409 |
| SYNTHETIC | L_only | 50 | 0 | 0.060 | 0.055 | 0.916 | 1.120 | 0.058 | 0.900 | 1.000 | 1.143 |
| SYNTHETIC | gate:primary | 50 | 0 | 0.020 | 0.056 | 0.890 | 1.136 | 0.081 | 0.900 | 0.625 | 0.988 |
| SYNTHETIC | gate:max_softmax | 48 | 2 | 0.021 | 0.059 | 0.897 | 1.165 | 0.083 | 0.905 | 0.708 | 1.083 |
| SYNTHETIC | gate:mean_logprob | 50 | 0 | 0.020 | 0.056 | 0.890 | 1.136 | 0.086 | 0.902 | 0.625 | 0.991 |
| SYNTHETIC | gate:entropy | 50 | 0 | 0.020 | 0.056 | 0.890 | 1.144 | 0.081 | 0.897 | 0.625 | 0.991 |
| SYNTHETIC | gate:margin | 50 | 0 | 0.000 | 0.057 | 0.893 | 1.157 | 0.081 | 0.900 | 0.625 | 0.988 |
| SYNTHETIC | random_router | 50 | 0 | 0.160 | 0.072 | 0.916 | 1.076 | 0.075 | 0.905 | 0.950 | 1.113 |
| SYNTHETIC | query_router | 50 | 0 | 0.040 | 0.058 | 0.909 | 1.105 | 0.058 | 0.900 | 1.000 | 1.143 |
| SYNTHETIC | self_consistency | 50 | 0 | 0.040 | 0.046 | 0.862 | 2.715 | 0.055 | 0.827 | 1.000 | 2.828 |

### signal_auroc

| synthetic | tier | signal | auroc | lo | hi |
|---|---|---|---|---|---|
| SYNTHETIC | S | max_prob | 0.868 | 0.829 | 0.905 |
| SYNTHETIC | S | first_margin | 0.863 | 0.824 | 0.901 |
| SYNTHETIC | S | neg_first_entropy | 0.865 | 0.826 | 0.902 |
| SYNTHETIC | S | mean_logprob | 0.884 | 0.847 | 0.919 |
| SYNTHETIC | S | min_logprob | 0.868 | 0.829 | 0.905 |
| SYNTHETIC | S | mean_margin | 0.877 | 0.839 | 0.912 |
| SYNTHETIC | S | neg_mean_entropy | 0.880 | 0.843 | 0.914 |
| SYNTHETIC | L | max_prob | 0.869 | 0.802 | 0.922 |
| SYNTHETIC | L | first_margin | 0.868 | 0.801 | 0.921 |
| SYNTHETIC | L | neg_first_entropy | 0.869 | 0.802 | 0.922 |
| SYNTHETIC | L | mean_logprob | 0.884 | 0.823 | 0.932 |
| SYNTHETIC | L | min_logprob | 0.869 | 0.802 | 0.922 |
| SYNTHETIC | L | mean_margin | 0.882 | 0.820 | 0.930 |
| SYNTHETIC | L | neg_mean_entropy | 0.883 | 0.822 | 0.931 |

### comparisons_gate_minus_other

| synthetic | comparison | acc_on_answered_diff | cost_per_correct_diff | coverage_diff | mcnemar_p |
|---|---|---|---|---|---|
| SYNTHETIC | vs_L_only | -0.022 [-0.039, -0.008] | -0.154 [-0.206, -0.097] | 0.000 [0.000, 0.000] | 0.008 |
| SYNTHETIC | vs_S_only | 0.011 [-0.024, 0.048] | 0.579 [0.526, 0.633] | 0.328 [0.285, 0.370] | 0.000 |
| SYNTHETIC | vs_gate:max_softmax | 0.002 [-0.009, 0.015] | -0.094 [-0.127, -0.066] | -0.005 [-0.015, 0.005] | 1.000 |
| SYNTHETIC | vs_gate:mean_logprob | 0.005 [0.000, 0.013] | -0.003 [-0.010, 0.000] | -0.002 [-0.012, 0.005] | 1.000 |
| SYNTHETIC | vs_gate:entropy | 0.000 [0.000, 0.001] | -0.003 [-0.010, 0.000] | 0.003 [0.000, 0.008] | 1.000 |
| SYNTHETIC | vs_gate:margin | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| SYNTHETIC | vs_random_router_certified | -0.006 [-0.028, 0.016] | -0.125 [-0.183, -0.068] | -0.005 [-0.013, 0.000] | 0.454 |
| SYNTHETIC | vs_query_router_certified | -0.022 [-0.038, -0.008] | -0.154 [-0.211, -0.101] | 0.000 [0.000, 0.000] | 0.008 |
| SYNTHETIC | vs_random_router@matched | 0.099 [0.060, 0.137] | 0.170 [0.099, 0.246] | -0.017 [-0.045, 0.008] | 0.000 |
| SYNTHETIC | vs_query_router@matched | 0.035 [0.010, 0.062] | 0.157 [0.096, 0.225] | -0.012 [-0.025, -0.002] | 0.115 |

### shift

| synthetic | dataset | graded | n | violation_freq | mean_risk | risk_ci95_split0 | drift_alarm_rate | n_windows |
|---|---|---|---|---|---|---|---|---|
| SYNTHETIC | synthetic-nq_open | yes | 300 | 0.020 | 0.077 | 0.087 [0.054, 0.126] | 0.167 | 6 |
| SYNTHETIC | synthetic-simpleqa_verified | yes | 200 | 0.740 | 0.134 | 0.143 [0.071, 0.222] | 1.000 | 4 |

### quantization

| synthetic | config | transferred_violation_freq | recalibrated_violation_freq | delta_auroc | delta_ece | auroc_equivalent | ece_equivalent |
|---|---|---|---|---|---|---|---|
| SYNTHETIC | synthetic-ladder_q8 | 0.040 | 0.060 | -0.004 [-0.029, 0.021] | -0.005 [-0.029, 0.039] | no | no |
| SYNTHETIC | synthetic-ladder_q4km | 0.040 | 0.020 | 0.005 [-0.017, 0.028] | -0.015 [-0.041, 0.034] | no | no |
| SYNTHETIC | synthetic-ladder_q3km | 0.040 | 0.040 | -0.005 [-0.043, 0.030] | 0.022 [-0.014, 0.064] | no | no |
| SYNTHETIC | synthetic-ladder_kvq4 | 0.040 | 0.040 | -0.002 [-0.030, 0.023] | 0.004 [-0.019, 0.050] | no | no |

### hypotheses

| synthetic | id | title | verdict | why |
|---|---|---|---|---|
| SYNTHETIC | H1 | validity | SUPPORTED | violation frequency 0.020, CP95 upper bound 0.106 <= 15% |
| SYNTHETIC | H2 | signal | NOT SUPPORTED | all single token signals within +-0.02 of max_prob |
| SYNTHETIC | H3 | cascade value | INCONCLUSIVE | not established: non-inferiority |
| SYNTHETIC | H4 | self-consistency | NOT SUPPORTED | no gain of >= 0.03 AUROC |
| SYNTHETIC | H5 | shift | NOT SUPPORTED | drift alarm rate 0.17 < 0.80 |
| SYNTHETIC | H6 | quantization | INCONCLUSIVE | neither the shift nor the equivalence criterion is met for all configs |
| SYNTHETIC | H7 | hard sets | NOT SUPPORTED | AUROC 0.899 |
