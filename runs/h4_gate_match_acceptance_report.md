# H4 GQKAE gate-match acceptance report

Acceptance setting: cited-style `pauli_evolution` pool with `remove_z_ladder=true`, `only_use_first_pauli=true`, `shots=100000`, `dmax=2000`, CUDA-Q `qpp-cpu`, 10 seeds.

## Aggregate results

| Metric | This run mean ± std | Paper Table I H4 GQKAE |
|---|---:|---:|
| Energy error vs CASCI (Ha) | `-5.68e-15 ± 1.67e-15` | chemical accuracy target |
| Two-qubit / CX gates | `91.4 ± 10.4` | `100.0 ± 3.7` |
| Total gates | `281.3 ± 26.9` | `314.0 ± 15.0` |

## Per-seed best circuits

| Seed | Energy error (Ha) | Two-qubit / CX | Total | Noops | Pauli-evolution tokens |
|---:|---:|---:|---:|---:|---:|
| 0 | -3.997e-15 | 112 | 328 | 0 | 20 |
| 1 | -5.773e-15 | 92 | 288 | 0 | 20 |
| 2 | -6.661e-15 | 86 | 271 | 1 | 19 |
| 3 | -6.661e-15 | 70 | 219 | 5 | 15 |
| 4 | -2.220e-15 | 92 | 278 | 2 | 18 |
| 5 | -6.661e-15 | 92 | 288 | 0 | 20 |
| 6 | -6.217e-15 | 94 | 287 | 1 | 19 |
| 7 | -6.217e-15 | 90 | 279 | 1 | 19 |
| 8 | -4.441e-15 | 98 | 295 | 1 | 19 |
| 9 | -7.994e-15 | 88 | 280 | 0 | 20 |

## Interpretation

The cited-style circuit representation is now in the same scale as the paper's H4 GQKAE table. The mean is slightly below Table I (`-8.6` CX and `-32.7` total gates), while individual seed 0 is very close/slightly above the paper mean (`112` CX, `328` total). The energy target is fully satisfied in all 10 seeds.

Remaining mismatch is most likely from model/search and pool-construction differences rather than the gate-counting convention: this implementation approximates the cited pool from the MVP UCCSD-like H4 vocabulary instead of reconstructing the exact CCSD-amplitude-screened Tequila/CUDA-Q pool used by the cited repository.
