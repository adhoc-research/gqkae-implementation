# CUDA-Q H4 local feasibility benchmark

This is a feasibility benchmark, not a final exact-paper acceptance run.

Sequence: `[6, 5, 0, 12, 9, 1, 20, 10, 20, 20, 1, 2, 16, 15, 10, 14, 5, 14, 14, 3]`
Shot counts: `[100, 1000, 5000, 10000]`

## Targets
- `qpp-cpu` target=`qpp-cpu` option=`None`
- `nvidia` target=`nvidia` option=`None`

## Timing
| target | shots | setup | build | sample | resources | QSCI | total | energy err | L1 prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qpp-cpu | 100 | 0.003 | 0.017 | 1.231 | 0.081 | 0.000 | 1.333 | 4.697e-03 | 0.108 |
| qpp-cpu | 1000 | 0.002 | 0.013 | 0.044 | 0.082 | 0.000 | 0.143 | 3.618e-04 | 0.031 |
| qpp-cpu | 5000 | 0.002 | 0.015 | 0.512 | 0.083 | 0.000 | 0.613 | 2.292e-05 | 0.029 |
| qpp-cpu | 10000 | 0.002 | 0.013 | 0.815 | 0.081 | 0.000 | 0.912 | 2.292e-05 | 0.018 |
| nvidia | 100 | 0.003 | 0.015 | 0.049 | 0.067 | 0.000 | 0.134 | 4.697e-03 | 0.108 |
| nvidia | 1000 | 0.002 | 0.014 | 0.036 | 0.064 | 0.000 | 0.117 | 3.618e-04 | 0.031 |
| nvidia | 5000 | 0.003 | 0.014 | 0.042 | 0.073 | 0.000 | 0.133 | 2.292e-05 | 0.029 |
| nvidia | 10000 | 0.003 | 0.014 | 0.046 | 0.066 | 0.000 | 0.130 | 2.292e-05 | 0.018 |

## ETA projections
### qpp-cpu
- assumption `rate_scaled_from_largest_shots`:
  - one 100000-shot sequence: 8.3s
  - one exact iteration, batch 10: 1.4m
  - one exact 100-iteration trial: 2.3h
  - five exact trials: 11.5h
  - PES one seed over grid: 22.9h (projection only)
  - PES five trials/grid point: 4.8d (projection only)
- assumption `linear_fit_total_vs_shots`:
  - one 100000-shot sequence: 1.4s
  - one exact iteration, batch 10: 13.5s
  - one exact 100-iteration trial: 22.5m
  - five exact trials: 1.9h
  - PES one seed over grid: 3.8h (projection only)
  - PES five trials/grid point: 18.8h (projection only)
- assumption `linear_fit_sample_vs_shots_plus_fixed`:
  - one 100000-shot sequence: 1.4s
  - one exact iteration, batch 10: 13.9s
  - one exact 100-iteration trial: 23.1m
  - five exact trials: 1.9h
  - PES one seed over grid: 3.8h (projection only)
  - PES five trials/grid point: 19.2h (projection only)
### nvidia
- assumption `rate_scaled_from_largest_shots`:
  - one 100000-shot sequence: 0.5s
  - one exact iteration, batch 10: 5.5s
  - one exact 100-iteration trial: 9.1m
  - five exact trials: 45.5m
  - PES one seed over grid: 1.5h (projection only)
  - PES five trials/grid point: 7.6h (projection only)
- assumption `linear_fit_total_vs_shots`:
  - one 100000-shot sequence: 0.2s
  - one exact iteration, batch 10: 1.7s
  - one exact 100-iteration trial: 2.9m
  - five exact trials: 14.4m
  - PES one seed over grid: 28.8m (projection only)
  - PES five trials/grid point: 2.4h (projection only)
- assumption `linear_fit_sample_vs_shots_plus_fixed`:
  - one 100000-shot sequence: 0.1s
  - one exact iteration, batch 10: 1.5s
  - one exact 100-iteration trial: 2.5m
  - five exact trials: 12.5m
  - PES one seed over grid: 25.0m (projection only)
  - PES five trials/grid point: 2.1h (projection only)

Exact five-trial and PES runs were **not run** in this phase and need user greenlight.
