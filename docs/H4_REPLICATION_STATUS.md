# H4 replication status

## Current status

**CUDA-Q H4 PES backend-fidelity evidence is now complete.**

The exact CUDA-Q `nvidia` run completed the paper H4 PES grid with five independent trials per bond length using exact paper settings:

- CUDA-Q target: `nvidia`
- shots: `100000`
- iterations: `100`
- batch circuits: `10`
- sequence length: `20`
- QSCI: `dmax=2000`, `symmetry_completion`
- seeds: `[0, 1, 2, 3, 4]`
- grid: `[0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30] Å`

## Key artifacts

- Final CUDA-Q PES report: `docs/CUDAQ_H4_PES_FIVE_TRIAL_REPORT.md`
- Aggregate JSON: `runs/h4_pes_cudaq_nvidia_summary.json`
- Aggregate runtime JSONL: `runs/h4_pes_cudaq_nvidia_runtime.jsonl`
- Aggregate markdown: `runs/h4_pes_cudaq_nvidia_report.md`
- Per-job artifacts: `runs/h4_pes_cudaq_nvidia_R*_seed*/`
- CUDA-Q feasibility report: `docs/CUDAQ_H4_FEASIBILITY_REPORT.md`

## CUDA-Q PES result

- Jobs completed: `50/50`
- Failures: `0`
- Total per-job runtime sum: `13092.9s` (`3.64h`)
- Chemical accuracy: `50/50` pass
- Mean absolute errors are at numerical precision for every bond length.

At the Table-I bond length `0.88 Å`, five CUDA-Q trials gave:

- CASCI: `-2.195180549992 Ha`
- mean GQKAE: `-2.195180549992 Ha`
- mean absolute error: `3.464e-15 Ha`
- mean CX/two-qubit gates: `100.4`
- mean total gates excluding HF x: `322.2`

Paper H4 Table-I target:

- CX/two-qubit gates: `100.0 ± 3.7`
- total gates: `314.0 ± 15.0`

The CUDA-Q five-trial result matches the paper resource scale under the accepted practical criterion and reaches chemical accuracy across the full PES grid.

## Conclusion

The previous backend caveat has been addressed for H4 PES: exact CUDA-Q `nvidia` execution was used for the full paper PES grid with five independent trials per grid point. The result supports the H4 replication claim using the paper-fidelity Tequila/CCSD operator pool, interleaved ordering, QSCI symmetry completion, CUDA-Q sampling, and paper gate-count convention.
