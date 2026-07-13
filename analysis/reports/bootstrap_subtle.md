# Environment bootstrap comparison (task-resample, n=172, 10000 resamples, sim-thr 0.5)

| environment | AnsAcc | 95% CI | Δ vs keep_all | P(env>base) |
|---|---|---|---|---|
| keep_all | 13.37 | [8.72,18.60] | — | — |
| oracle | 15.12 | [9.88,20.93] | +1.74 | 0.709 |
| call_frequency | 9.88 | [5.81,14.53] | -3.49 | 0.083 |
| ours_echo | 8.14 | [4.07,12.21] | -5.23 | 0.007 |
