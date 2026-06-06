# CUDA-Q H4 PES five-trial report

Exact CUDA-Q PES run over the H4 paper grid with five independent trials per bond length.

- target: `nvidia` option=`None`
- shots: `100000`
- iterations: `100`
- batch circuits: `10`
- total elapsed: `13092.9s`
- completed/skipped jobs: `50/50`
- failures: `0`
- all chemical accuracy: `True`

## PES energy table
| R (Å) | n | CASCI (Ha) | mean GQKAE (Ha) | std GQKAE | mean abs error (Ha) | chem pass | mean CX | mean total excl. HF x | mean runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 5 | -1.766342826455 | -1.766342826455 | 6.822e-16 | 3.642e-15 | 1.00 | 94.8 | 303.4 | 223.5s |
| 0.60 | 5 | -2.015943326612 | -2.015943326612 | 5.179e-16 | 3.908e-15 | 1.00 | 101.6 | 322.0 | 225.3s |
| 0.70 | 5 | -2.133735251270 | -2.133735251270 | 6.647e-16 | 3.375e-15 | 1.00 | 99.2 | 316.4 | 241.5s |
| 0.80 | 5 | -2.183129188255 | -2.183129188255 | 4.529e-16 | 5.151e-15 | 1.00 | 103.6 | 322.2 | 248.8s |
| 0.88 | 5 | -2.195180549992 | -2.195180549992 | 7.105e-16 | 3.464e-15 | 1.00 | 100.4 | 322.2 | 262.1s |
| 0.90 | 5 | -2.195712227345 | -2.195712227345 | 3.323e-16 | 3.642e-15 | 1.00 | 100.0 | 321.6 | 285.0s |
| 1.00 | 5 | -2.188673801759 | -2.188673801759 | 5.329e-16 | 5.951e-15 | 1.00 | 99.2 | 316.8 | 286.7s |
| 1.10 | 5 | -2.171714328371 | -2.171714328371 | 2.809e-16 | 8.882e-16 | 1.00 | 101.2 | 321.8 | 266.8s |
| 1.20 | 5 | -2.150179440655 | -2.150179440655 | 3.972e-16 | 7.550e-15 | 1.00 | 96.4 | 309.4 | 265.3s |
| 1.30 | 5 | -2.126998149910 | -2.126998149910 | 3.323e-16 | 4.796e-15 | 1.00 | 102.8 | 330.6 | 313.4s |

## Per-job rows
| R (Å) | seed | status | energy | abs error | CX | total excl. HF x | elapsed | run dir |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.50 | 0 | completed | -1.766342826455 | 3.109e-15 | 96.0 | 312.0 | 239.3s | `runs/h4_pes_cudaq_nvidia_R0p50_seed0` |
| 0.50 | 1 | completed | -1.766342826455 | 4.663e-15 | 88.0 | 282.0 | 216.3s | `runs/h4_pes_cudaq_nvidia_R0p50_seed1` |
| 0.50 | 2 | completed | -1.766342826455 | 3.331e-15 | 100.0 | 304.0 | 228.2s | `runs/h4_pes_cudaq_nvidia_R0p50_seed2` |
| 0.50 | 3 | completed | -1.766342826455 | 2.887e-15 | 100.0 | 324.0 | 218.8s | `runs/h4_pes_cudaq_nvidia_R0p50_seed3` |
| 0.50 | 4 | completed | -1.766342826455 | 4.219e-15 | 90.0 | 295.0 | 215.1s | `runs/h4_pes_cudaq_nvidia_R0p50_seed4` |
| 0.60 | 0 | completed | -2.015943326612 | 4.441e-15 | 102.0 | 327.0 | 215.6s | `runs/h4_pes_cudaq_nvidia_R0p60_seed0` |
| 0.60 | 1 | completed | -2.015943326612 | 3.109e-15 | 100.0 | 324.0 | 230.1s | `runs/h4_pes_cudaq_nvidia_R0p60_seed1` |
| 0.60 | 2 | completed | -2.015943326612 | 4.441e-15 | 100.0 | 312.0 | 223.7s | `runs/h4_pes_cudaq_nvidia_R0p60_seed2` |
| 0.60 | 3 | completed | -2.015943326612 | 3.997e-15 | 90.0 | 295.0 | 222.5s | `runs/h4_pes_cudaq_nvidia_R0p60_seed3` |
| 0.60 | 4 | completed | -2.015943326612 | 3.553e-15 | 116.0 | 352.0 | 234.7s | `runs/h4_pes_cudaq_nvidia_R0p60_seed4` |
| 0.70 | 0 | completed | -2.133735251270 | 3.553e-15 | 94.0 | 303.0 | 243.9s | `runs/h4_pes_cudaq_nvidia_R0p70_seed0` |
| 0.70 | 1 | completed | -2.133735251270 | 3.553e-15 | 104.0 | 340.0 | 239.2s | `runs/h4_pes_cudaq_nvidia_R0p70_seed1` |
| 0.70 | 2 | completed | -2.133735251270 | 2.665e-15 | 96.0 | 304.0 | 243.6s | `runs/h4_pes_cudaq_nvidia_R0p70_seed2` |
| 0.70 | 3 | completed | -2.133735251270 | 4.441e-15 | 98.0 | 315.0 | 244.3s | `runs/h4_pes_cudaq_nvidia_R0p70_seed3` |
| 0.70 | 4 | completed | -2.133735251270 | 2.665e-15 | 104.0 | 320.0 | 236.7s | `runs/h4_pes_cudaq_nvidia_R0p70_seed4` |
| 0.80 | 0 | completed | -2.183129188255 | 5.329e-15 | 102.0 | 315.0 | 251.2s | `runs/h4_pes_cudaq_nvidia_R0p80_seed0` |
| 0.80 | 1 | completed | -2.183129188255 | 4.885e-15 | 104.0 | 330.0 | 248.8s | `runs/h4_pes_cudaq_nvidia_R0p80_seed1` |
| 0.80 | 2 | completed | -2.183129188255 | 5.329e-15 | 102.0 | 307.0 | 248.0s | `runs/h4_pes_cudaq_nvidia_R0p80_seed2` |
| 0.80 | 3 | completed | -2.183129188255 | 5.773e-15 | 98.0 | 315.0 | 247.1s | `runs/h4_pes_cudaq_nvidia_R0p80_seed3` |
| 0.80 | 4 | completed | -2.183129188255 | 4.441e-15 | 112.0 | 344.0 | 249.2s | `runs/h4_pes_cudaq_nvidia_R0p80_seed4` |
| 0.88 | 0 | completed | -2.195180549992 | 3.109e-15 | 104.0 | 328.0 | 256.8s | `runs/h4_pes_cudaq_nvidia_R0p88_seed0` |
| 0.88 | 1 | completed | -2.195180549992 | 3.109e-15 | 100.0 | 320.0 | 259.4s | `runs/h4_pes_cudaq_nvidia_R0p88_seed1` |
| 0.88 | 2 | completed | -2.195180549992 | 4.885e-15 | 94.0 | 315.0 | 259.9s | `runs/h4_pes_cudaq_nvidia_R0p88_seed2` |
| 0.88 | 3 | completed | -2.195180549992 | 3.109e-15 | 108.0 | 340.0 | 263.7s | `runs/h4_pes_cudaq_nvidia_R0p88_seed3` |
| 0.88 | 4 | completed | -2.195180549992 | 3.109e-15 | 96.0 | 308.0 | 270.9s | `runs/h4_pes_cudaq_nvidia_R0p88_seed4` |
| 0.90 | 0 | completed | -2.195712227345 | 3.553e-15 | 102.0 | 319.0 | 281.3s | `runs/h4_pes_cudaq_nvidia_R0p90_seed0` |
| 0.90 | 1 | completed | -2.195712227345 | 3.553e-15 | 104.0 | 332.0 | 279.0s | `runs/h4_pes_cudaq_nvidia_R0p90_seed1` |
| 0.90 | 2 | completed | -2.195712227345 | 3.997e-15 | 92.0 | 298.0 | 289.2s | `runs/h4_pes_cudaq_nvidia_R0p90_seed2` |
| 0.90 | 3 | completed | -2.195712227345 | 3.997e-15 | 108.0 | 352.0 | 288.4s | `runs/h4_pes_cudaq_nvidia_R0p90_seed3` |
| 0.90 | 4 | completed | -2.195712227345 | 3.109e-15 | 94.0 | 307.0 | 287.2s | `runs/h4_pes_cudaq_nvidia_R0p90_seed4` |
| 1.00 | 0 | completed | -2.188673801759 | 6.217e-15 | 116.0 | 352.0 | 288.2s | `runs/h4_pes_cudaq_nvidia_R1p00_seed0` |
| 1.00 | 1 | completed | -2.188673801759 | 6.661e-15 | 88.0 | 282.0 | 285.0s | `runs/h4_pes_cudaq_nvidia_R1p00_seed1` |
| 1.00 | 2 | completed | -2.188673801759 | 6.217e-15 | 108.0 | 348.0 | 288.2s | `runs/h4_pes_cudaq_nvidia_R1p00_seed2` |
| 1.00 | 3 | completed | -2.188673801759 | 5.329e-15 | 90.0 | 291.0 | 290.4s | `runs/h4_pes_cudaq_nvidia_R1p00_seed3` |
| 1.00 | 4 | completed | -2.188673801759 | 5.329e-15 | 94.0 | 311.0 | 281.8s | `runs/h4_pes_cudaq_nvidia_R1p00_seed4` |
| 1.10 | 0 | completed | -2.171714328371 | 8.882e-16 | 108.0 | 336.0 | 276.3s | `runs/h4_pes_cudaq_nvidia_R1p10_seed0` |
| 1.10 | 1 | completed | -2.171714328371 | 1.332e-15 | 108.0 | 340.0 | 287.6s | `runs/h4_pes_cudaq_nvidia_R1p10_seed1` |
| 1.10 | 2 | completed | -2.171714328371 | 8.882e-16 | 96.0 | 318.0 | 262.0s | `runs/h4_pes_cudaq_nvidia_R1p10_seed2` |
| 1.10 | 3 | completed | -2.171714328371 | 8.882e-16 | 92.0 | 292.0 | 251.4s | `runs/h4_pes_cudaq_nvidia_R1p10_seed3` |
| 1.10 | 4 | completed | -2.171714328371 | 4.441e-16 | 102.0 | 323.0 | 256.4s | `runs/h4_pes_cudaq_nvidia_R1p10_seed4` |
| 1.20 | 0 | completed | -2.150179440655 | 7.550e-15 | 108.0 | 336.0 | 256.2s | `runs/h4_pes_cudaq_nvidia_R1p20_seed0` |
| 1.20 | 1 | completed | -2.150179440655 | 7.105e-15 | 98.0 | 311.0 | 245.6s | `runs/h4_pes_cudaq_nvidia_R1p20_seed1` |
| 1.20 | 2 | completed | -2.150179440655 | 7.105e-15 | 82.0 | 265.0 | 259.0s | `runs/h4_pes_cudaq_nvidia_R1p20_seed2` |
| 1.20 | 3 | completed | -2.150179440655 | 7.994e-15 | 100.0 | 332.0 | 269.4s | `runs/h4_pes_cudaq_nvidia_R1p20_seed3` |
| 1.20 | 4 | completed | -2.150179440655 | 7.994e-15 | 94.0 | 303.0 | 296.3s | `runs/h4_pes_cudaq_nvidia_R1p20_seed4` |
| 1.30 | 0 | completed | -2.126998149910 | 4.885e-15 | 108.0 | 340.0 | 323.0s | `runs/h4_pes_cudaq_nvidia_R1p30_seed0` |
| 1.30 | 1 | completed | -2.126998149910 | 4.441e-15 | 94.0 | 335.0 | 318.2s | `runs/h4_pes_cudaq_nvidia_R1p30_seed1` |
| 1.30 | 2 | completed | -2.126998149910 | 4.441e-15 | 108.0 | 332.0 | 320.1s | `runs/h4_pes_cudaq_nvidia_R1p30_seed2` |
| 1.30 | 3 | completed | -2.126998149910 | 4.885e-15 | 102.0 | 323.0 | 309.2s | `runs/h4_pes_cudaq_nvidia_R1p30_seed3` |
| 1.30 | 4 | completed | -2.126998149910 | 5.329e-15 | 102.0 | 323.0 | 296.7s | `runs/h4_pes_cudaq_nvidia_R1p30_seed4` |

## Command
```bash
uv run --extra paper --extra cuda python scripts/run_h4_cudaq_pes_five_trials.py --config configs/h4_paper_fidelity_cudaq_nvidia.yaml --seeds 0 1 2 3 4 --target nvidia --output-prefix runs/h4_pes_cudaq_nvidia --skip-existing
```
