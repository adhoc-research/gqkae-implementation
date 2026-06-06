# CUDA-Q H4 local feasibility report

This phase is a local feasibility exploration, **not** a final exact-paper CUDA-Q acceptance run.

## Backend discovery

Artifact: `runs/cudaq_target_diagnostics.json`

- CUDA-Q: `CUDA-Q Version 0.14.2`
- Random seed API: available via `cudaq.set_random_seed`
- GPU visible through `nvidia-smi`:
  - `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, 12227 MiB, driver `596.49`
- Usable targets:
  - CPU baseline: `qpp-cpu`
  - GPU: `nvidia`
  - GPU options also smoke-usable: `nvidia option=fp64`, `nvidia option=fp32`
- Unusable candidate names on this install:
  - `qpp-cuda`
  - `custatevec`

The benchmark auto-selected `nvidia` as the GPU target.

## Fixed H4 benchmark setup

Artifact: `runs/cudaq_h4_target_benchmark.json`  
Readable summary: `runs/cudaq_h4_target_benchmark.md`

Benchmark sequence:

```text
[6, 5, 0, 12, 9, 1, 20, 10, 20, 20, 1, 2, 16, 15, 10, 14, 5, 14, 14, 3]
```

The benchmark uses H4 0.88 Å, the Tequila-backed paper-fidelity pool, interleaved ordering, `dmax=2000`, and QSCI `symmetry_completion`. Shot counts tested: `[100, 1000, 5000, 10000]`.

## Timing summary

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

Interpretation:

- `nvidia` is consistently faster than `qpp-cpu` for this fixed-sequence workload.
- The GPU advantage at 10000 shots is roughly `0.912 / 0.130 ≈ 7.0x` on total fixed-sequence time and `0.815 / 0.046 ≈ 17.7x` on sampling time.
- QSCI solve time is negligible for H4 CAS(4,4), because the determinant space is only 36 determinants.
- CUDA-Q sampled probabilities agree with full-qubit deterministic probabilities within the 5σ finite-shot binomial check at every tested shot count and both targets.
- Gate resources match the paper/static convention for the fixed sequence: `two_qubit=98`, `total_including_hf_x=307`, `total_excluding_hf_x=303`.

## ETA projections for exact paper Table-I settings

Projection target settings:

- one sequence: `100000` shots
- one iteration: `batch_circuits=10`
- one full trial: `100` iterations
- paper-style comparison target: **five independent trials**

These are projections from a fixed-sequence benchmark, not completed training runs. They primarily estimate CUDA-Q sampling/resource/QSCI workload and may not include all neural-policy overhead.

### Conservative projection: scale from largest shot count

| target | one 100k-shot sequence | one iteration | one 100-iteration trial | five trials |
|---|---:|---:|---:|---:|
| qpp-cpu | 8.3s | 1.4m | 2.3h | 11.5h |
| nvidia | 0.5s | 5.5s | 9.1m | 45.5m |

### Linear-fit projections

The linear-fit projections are less conservative, especially for GPU where fixed overhead dominates at these small circuits.

| target | assumption | one 100k-shot sequence | one trial | five trials |
|---|---|---:|---:|---:|
| qpp-cpu | total-vs-shots fit | 1.4s | 22.5m | 1.9h |
| qpp-cpu | sample-fit + fixed | 1.4s | 23.1m | 1.9h |
| nvidia | total-vs-shots fit | 0.2s | 2.9m | 14.4m |
| nvidia | sample-fit + fixed | 0.1s | 2.5m | 12.5m |

Recommended planning range:

- CPU exact five-trial Table-I: approximately `2h` to `12h`.
- GPU exact five-trial Table-I: approximately `15m` to `46m`.

The conservative GPU estimate is the safer next-phase planning number.

## ETA projections for exact CUDA-Q H4 PES

Extracted PES grid has 10 bond lengths. PES projections below assume the same exact-paper workload per grid point. These are **projections only**; no exact CUDA-Q PES was run in this phase.

### Conservative projection: scale from largest shot count

| target | one seed over full grid | five trials per grid point |
|---|---:|---:|
| qpp-cpu | 22.9h | 4.8d |
| nvidia | 1.5h | 7.6h |

### Linear-fit projection range

| target | one seed over full grid | five trials per grid point |
|---|---:|---:|
| qpp-cpu | 3.8h | 18.8–19.2h |
| nvidia | 25–29m | 2.1–2.4h |

Recommended planning range:

- GPU PES, one seed over grid: approximately `30m` to `1.5h`.
- GPU PES, five trials per grid point: approximately `2.5h` to `8h`.
- CPU PES is likely inconvenient locally unless run overnight or in pieces.

## Numerical/backend consistency

Fixed-sequence checks performed in the benchmark:

- Full-qubit deterministic probability reference vs CUDA-Q sampled counts.
- Bitstring orientation under `cudaq_reverse_bitstrings=false`.
- QSCI determinant subspace and energy from sampled CUDA-Q counts.
- Paper/static gate resource checks.

Results:

- Both `qpp-cpu` and `nvidia` produce identical sampled top-level behavior for the fixed sequence under the same seed.
- No probability bin exceeded the 5σ finite-shot tolerance when compared to the full-qubit deterministic reference.
- QSCI energy improves with shots and is chemically accurate by 1000 shots for this sequence; at 10000 shots error is `2.292e-05 Ha`.

## Recommended next-phase configs, not run here

Exact five-trial Table-I, GPU target:

```bash
uv run --extra paper --extra cuda python scripts/run_h4_paper_fidelity_sweep.py \
  --config configs/h4_paper_fidelity.yaml \
  --backend cudaq \
  --output-prefix runs/h4_paper_fidelity_cudaq_nvidia_seed \
  --seeds 0 1 2 3 4 \
  --skip-existing
```

with config overrides or a copied config setting:

```yaml
qsci:
  backend: cudaq
  cudaq_target: nvidia
  cudaq_option: null
  shots: 100000
training:
  iterations: 100
  batch_circuits: 10
```

Exact CUDA-Q PES, GPU target, one seed:

```bash
uv run --extra paper --extra cuda python scripts/run_h4_pes.py \
  --config configs/h4_paper_fidelity.yaml \
  --backend cudaq \
  --seed 7 \
  --output-prefix runs/h4_pes_paper_fidelity_cudaq_nvidia \
  --skip-existing
```

Before running PES with the current script, ensure the config sets `qsci.cudaq_target: nvidia`; the script currently overrides only `qsci.backend`.

## Status

The CUDA-Q backend caveat is **not removed** by this feasibility phase. To remove it under the user's stated criteria, exact CUDA-Q Table-I and exact CUDA-Q PES evidence must be run after the user selects and greenlights a target/config. Based on this benchmark, `nvidia` appears locally feasible and substantially faster than `qpp-cpu`.
