# Paper Comparison With AutoSSVH And Ours

- CSV: `/media/kejunjie_coco/autohash/autossh_newwek/BPM-PGAFS+SHVD+PCMR/eval_results/paper_comparison_with_autossvh_ours_20260625.csv`
- AutoSSVH rows use the values provided by the user; Avg is the arithmetic mean of the six displayed mAP metrics.
- Ours rows are the current selected/checkpoint results already merged in the previous table.

| Dataset | Bit | Method | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | Avg | Selected_param |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ActivityNet | 16 | AutoSSVH | 0.176000 | 0.091000 | 0.055000 | 0.040000 | 0.031000 | 0.026000 | 0.069833 | reported AutoSSVH result |
| ActivityNet | 16 | Ours | 0.183467 | 0.099024 | 0.060520 | 0.043424 | 0.033617 | 0.027473 | 0.074587 | r06_p045 (ratio=0.6, prob=0.45) |
| ActivityNet | 32 | AutoSSVH | 0.250000 | 0.136000 | 0.082000 | 0.061000 | 0.049000 | 0.038000 | 0.102667 | reported AutoSSVH result |
| ActivityNet | 32 | Ours | 0.251187 | 0.135583 | 0.080888 | 0.057204 | 0.044074 | 0.035905 | 0.100807 | r07_p035 (ratio=0.7, prob=0.35) |
| ActivityNet | 64 | AutoSSVH | 0.290000 | 0.164000 | 0.098000 | 0.071000 | 0.055000 | 0.045000 | 0.120500 | reported AutoSSVH result |
| ActivityNet | 64 | Ours | 0.306237 | 0.167668 | 0.100578 | 0.070788 | 0.054343 | 0.044105 | 0.123953 | r03_p025 (ratio=0.3, prob=0.25) |
| FCVID | 16 | AutoSSVH | 0.347000 | 0.256000 | 0.225000 | 0.207000 | 0.193000 | 0.180000 | 0.234667 | reported AutoSSVH result |
| FCVID | 16 | Ours | 0.378807 | 0.268343 | 0.231222 | 0.210029 | 0.194298 | 0.181256 | 0.243992 | r05_g005_sw002 (FCVID current trained setting; no 20x sweep) |
| FCVID | 32 | AutoSSVH | 0.483000 | 0.336000 | 0.289000 | 0.263000 | 0.244000 | 0.228000 | 0.307167 | reported AutoSSVH result |
| FCVID | 32 | Ours | 0.505501 | 0.351828 | 0.303271 | 0.276665 | 0.256572 | 0.239852 | 0.322282 | r05_g005_sw002 (FCVID current trained setting; no 20x sweep) |
| FCVID | 64 | AutoSSVH | 0.539000 | 0.381000 | 0.334000 | 0.309000 | 0.290000 | 0.273000 | 0.354333 | reported AutoSSVH result |
| FCVID | 64 | Ours | 0.544756 | 0.392097 | 0.342343 | 0.314552 | 0.292908 | 0.274387 | 0.360174 | r05_g005_sw002 (FCVID current trained setting; no 20x sweep) |
| UCF101 | 16 | AutoSSVH | 0.423000 | 0.343000 | 0.288000 | 0.244000 | 0.210000 | 0.184000 | 0.282000 | reported AutoSSVH result |
| UCF101 | 16 | Ours | 0.438561 | 0.366869 | 0.313106 | 0.266730 | 0.227931 | 0.197278 | 0.301746 | r06_p055 (ratio=0.6, prob=0.55) |
| UCF101 | 32 | AutoSSVH | 0.519000 | 0.448000 | 0.396000 | 0.353000 | 0.309000 | 0.269000 | 0.382333 | reported AutoSSVH result |
| UCF101 | 32 | Ours | 0.549745 | 0.475387 | 0.416246 | 0.366596 | 0.316500 | 0.272295 | 0.399462 | r06_p025 (ratio=0.6, prob=0.25) |
| UCF101 | 64 | AutoSSVH | 0.570000 | 0.500000 | 0.452000 | 0.413000 | 0.373000 | 0.329000 | 0.439500 | reported AutoSSVH result |
| UCF101 | 64 | Ours | 0.590889 | 0.517527 | 0.460946 | 0.412354 | 0.363179 | 0.316538 | 0.443572 | r06_p055 (ratio=0.6, prob=0.55) |
| HMDB51 | 16 | AutoSSVH | 0.161000 | 0.113000 | 0.081000 | 0.064000 | 0.053000 | 0.045000 | 0.086167 | reported AutoSSVH result |
| HMDB51 | 16 | Ours | 0.179891 | 0.110484 | 0.077407 | 0.058837 | 0.047496 | 0.040777 | 0.085815 | r06_p045 (ratio=0.6, prob=0.45) |
| HMDB51 | 32 | AutoSSVH | 0.227000 | 0.163000 | 0.123000 | 0.097000 | 0.081000 | 0.069000 | 0.126667 | reported AutoSSVH result |
| HMDB51 | 32 | Ours | 0.225887 | 0.151340 | 0.108049 | 0.083229 | 0.067408 | 0.056606 | 0.115420 | r05_p045 (ratio=0.5, prob=0.45) |
| HMDB51 | 64 | AutoSSVH | 0.256000 | 0.175000 | 0.137000 | 0.111000 | 0.090000 | 0.076000 | 0.140833 | reported AutoSSVH result |
| HMDB51 | 64 | Ours | 0.259715 | 0.181656 | 0.136906 | 0.108749 | 0.089426 | 0.075467 | 0.141987 | r05_p055 (ratio=0.5, prob=0.55) |
