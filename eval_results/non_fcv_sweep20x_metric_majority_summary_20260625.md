# Non-FCV sweep20x metric-majority summary

Selection rule: metric_win_count desc, avg_metric_rank asc, mean_relative_to_best desc, mAP@5 desc.

| dataset | bit | selected combo | ratio | prob | metric wins | win list | avg rank | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| activitynet | 16 | r06_p045 | 0.6 | 0.45 | 5 | mAP@20;mAP@40;mAP@60;mAP@80;mAP@100 | 1.33 | 0.183467 | 0.099024 | 0.060520 | 0.043424 | 0.033617 | 0.027473 |
| activitynet | 32 | r07_p035 | 0.7 | 0.35 | 3 | mAP@60;mAP@80;mAP@100 | 2.50 | 0.251187 | 0.135583 | 0.080888 | 0.057204 | 0.044074 | 0.035905 |
| activitynet | 64 | r03_p025 | 0.3 | 0.25 | 5 | mAP@20;mAP@40;mAP@60;mAP@80;mAP@100 | 1.17 | 0.306237 | 0.167668 | 0.100578 | 0.070788 | 0.054343 | 0.044105 |
| hmdb | 16 | r06_p045 | 0.6 | 0.45 | 4 | mAP@5;mAP@20;mAP@40;mAP@60 | 1.33 | 0.179891 | 0.110484 | 0.077407 | 0.058837 | 0.047496 | 0.040777 |
| hmdb | 32 | r05_p045 | 0.5 | 0.45 | 5 | mAP@20;mAP@40;mAP@60;mAP@80;mAP@100 | 1.33 | 0.225887 | 0.151340 | 0.108049 | 0.083229 | 0.067408 | 0.056606 |
| hmdb | 64 | r05_p055 | 0.5 | 0.55 | 5 | mAP@20;mAP@40;mAP@60;mAP@80;mAP@100 | 1.33 | 0.259715 | 0.181656 | 0.136906 | 0.108749 | 0.089426 | 0.075467 |
| ucf | 16 | r06_p055 | 0.6 | 0.55 | 4 | mAP@40;mAP@60;mAP@80;mAP@100 | 1.83 | 0.438561 | 0.366869 | 0.313106 | 0.266730 | 0.227931 | 0.197278 |
| ucf | 32 | r06_p025 | 0.6 | 0.25 | 2 | mAP@60;mAP@100 | 2.33 | 0.549745 | 0.475387 | 0.416246 | 0.366596 | 0.316500 | 0.272295 |
| ucf | 64 | r06_p055 | 0.6 | 0.55 | 6 | mAP@5;mAP@20;mAP@40;mAP@60;mAP@80;mAP@100 | 1.00 | 0.590889 | 0.517527 | 0.460946 | 0.412354 | 0.363179 | 0.316538 |

Incomplete settings: UCF 32-bit has 17/20 evaluated rows in current CSV; selection is based on evaluated rows only.