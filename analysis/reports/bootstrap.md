# Environment bootstrap comparison (task-resample, n=172, 10000 resamples, sim-thr 0.5)

| environment | AnsAcc | 95% CI | Δ vs keep_all | P(env>base) |
|---|---|---|---|---|
| keep_all | 7.56 | [4.07,11.63] | — | — |
| oracle | 15.12 | [9.88,20.93] | +7.56 | 0.996 |
| random | 11.63 | [6.98,16.86] | +4.07 | 0.935 |
| call_frequency | 18.60 | [12.79,24.42] | +11.05 | 1.000 |
| error_rate | 8.72 | [4.65,13.37] | +1.16 | 0.682 |
| ours_echo | 16.86 | [11.63,22.67] | +9.30 | 0.999 |
