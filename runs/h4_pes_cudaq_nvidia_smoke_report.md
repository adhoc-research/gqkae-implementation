# CUDA-Q H4 PES five-trial report

Exact CUDA-Q PES run over the H4 paper grid with five independent trials per bond length.

- target: `nvidia` option=`None`
- shots: `1000`
- iterations: `1`
- batch circuits: `1`
- total elapsed: `20.4s`
- completed/skipped jobs: `1/1`
- failures: `0`
- all chemical accuracy: `False`

## PES energy table
| R (Å) | n | CASCI (Ha) | mean GQKAE (Ha) | std GQKAE | mean |error| (Ha) | chem pass | mean CX | mean total excl. HF x | mean runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.88 | 1 | -2.195180549992 | -2.185158361669 | 0.000e+00 | 1.002e-02 | 0.00 | 106.0 | 335.0 | 31.6s |

## Per-job rows
| R (Å) | seed | status | energy | |error| | CX | total excl. HF x | elapsed | run dir |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.88 | 0 | skipped_existing | -2.185158361669 | 1.002e-02 | 106.0 | 335.0 | 31.6s | `runs/h4_pes_cudaq_nvidia_smoke_R0p88_seed0` |

## Command
```bash
scripts/run_h4_cudaq_pes_five_trials.py --config configs/h4_paper_fidelity_cudaq_nvidia.yaml --bond-grid 0.88 --seeds 0 --target nvidia --output-prefix runs/h4_pes_cudaq_nvidia_smoke --override qsci.shots=1000 --override training.iterations=1 --override training.batch_circuits=1 --override training.policy_updates=1 --detailed-timing --skip-existing --stop-on-failure
```
