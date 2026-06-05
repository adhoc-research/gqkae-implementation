# H4 GQKAE paper gate-count matching experiments

## Context

The exact-counting branch now reproduces the cited Pauli-evolution counting convention, but the current MVP H4 paper-like run still reports much larger circuits than Table I:

| Source | Two-qubit / CX | Total gates | Notes |
|---|---:|---:|---|
| Paper Table I H4 GQKAE | `100.0 ± 3.7` | `314.0 ± 15.0` | all-to-all, arbitrary rotations + CX + Clifford gates |
| Current `runs/h4_cudaq_paper_like` | `1184` | `2728` | full JW expansion of each excitation token |
| Same best sequence, static cited-like reduction | `96` | `296` | remove Z ladder + only first Pauli term per excitation |

The last row is the key clue: the cited implementation's default config uses a `pauli_evolution` pool with `remove_z_ladder: true` and `only_use_first_pauli: true`. Applying those reductions to the current best sequence lands very close to the paper's H4 GQKAE scale.

## Approach

Run a controlled experiment series that separates three effects:

1. **Representation effect**: full excitation operators vs single Pauli-evolution tokens.
2. **Z-ladder effect**: Jordan-Wigner strings retained vs removed from Pauli words.
3. **Training/search effect**: whether the HQKANsformer discovers low-energy sequences whose counts match the paper once the operator pool matches the cited implementation.

The recommended implementation path for the experiments is to add an explicit cited-style `pauli_evolution` operator-pool mode and keep the current full-excitation mode as the baseline. Each experiment should write a compact result record with energy error, best sequence, vocab size, token composition, CX count, total count, and run command.

## Files to modify

- `src/gqkae/config.py`
  - Add operator-pool knobs: `spec`, `ccsd_threshold`, `remove_z_ladder`, `only_use_first_pauli`, and possibly `dedupe_pauli_words`.
- `src/gqkae/operator_pool.py`
  - Add a Pauli-evolution token type/pool that mirrors the cited `PauliEvolutionPool` behavior.
  - Preserve current UCCSD excitation pool as `spec: excitation`.
- `src/gqkae/fermion_mapping.py`
  - Reuse existing JW terms; add helper(s) to strip Z ladder and select one Pauli term deterministically.
- `src/gqkae/circuits.py`
  - Ensure sequence simulation/source/counting handles both excitation tokens and Pauli-evolution tokens.
- `src/gqkae/gate_counting.py`
  - Reuse the existing paper-style Pauli-word counter for the new pool mode.
- `src/gqkae/training/runner.py`
  - Add summary fields for pool spec, token composition, and paper-target deltas.
- `configs/`
  - Add experiment configs for the matrix below.
- `scripts/`
  - Add a small experiment runner/aggregator, e.g. `scripts/run_h4_gate_match_experiments.py` and/or `scripts/summarize_h4_gate_match.py`.
- `runs/`
  - Commit selected final artifacts after the experiment sweep.

## Reuse

- `src/gqkae/gate_counting.py::paper_style_pauli_evolution_gate_count`
- `src/gqkae/gate_counting.py::paper_style_sequence_gate_count`
- `src/gqkae/fermion_mapping.py::excitation_pauli_terms`
- `src/gqkae/circuits.py::sequence_to_cudaq_source`
- `src/gqkae/training/runner.py::train`
- Cited implementation behavior:
  - `gqe_qsci/gqe/operator_pool.py::PauliEvolutionPool`
  - defaults: `operator_pool.spec: pauli_evolution`, `ccsd_threshold: 1e-6`, `remove_z_ladder: true`, `only_use_first_pauli: true`

## Experiment matrix

### Phase 0: diagnostics, no retraining

- [ ] Recount existing best sequences under variants:
  - full excitation terms
  - full terms with Z ladder removed
  - first Pauli term only
  - first Pauli term with Z ladder removed
- [ ] Report per-token count table for all H4 vocabulary tokens.
- [ ] Confirm current best sequence under `first_pauli + remove_z_ladder` stays near `96 / 296`.

### Phase 1: cited-style pool implementation smoke tests

- [ ] `E1-baseline-excitation-full`: current full excitation mode, short run, confirms old high-count baseline.
- [ ] `E2-excitation-first-pauli-no-z`: current excitation vocabulary but executes/counts only first stripped Pauli term per selected excitation.
- [ ] `E3-pauli-evolution-cited-default`: cited-like Pauli-evolution vocabulary, `remove_z_ladder=true`, `only_use_first_pauli=true`, `ccsd_threshold=1e-6`.

### Phase 2: training sweeps toward Table I

Run each with at least 3 seeds initially; expand to 10 seeds only for promising settings.

| ID | Pool | L | Threshold | Z ladder | First Pauli | Expected count scale |
|---|---|---:|---:|---|---|---|
| A | excitation | 20 | n/a | keep | all | `~1000+ / ~2500+` baseline |
| B | excitation reduced | 20 | n/a | remove | yes | `~90-110 / ~290-320` |
| C | pauli_evolution | 20 | `1e-6` | remove | yes | closest cited-default reproduction |
| D | pauli_evolution | 20 | `1e-5` | remove | yes | smaller vocab / possibly lower counts |
| E | pauli_evolution | 20 | `1e-4` | remove | yes | aggressive screening, risk to energy |
| F | pauli_evolution | 22 | `1e-6` | remove | yes | test if total-gate mean moves toward `314` |
| G | pauli_evolution | 24 | `1e-6` | remove | yes | test sequence-composition/count sensitivity |

### Phase 3: paper-comparison acceptance run

- [ ] Pick the best setting satisfying energy accuracy and closest count match.
- [ ] Run 10 seeds with `shots=100000`, `dmax=2000`, CUDA-Q target available locally.
- [ ] Report mean ± std for:
  - energy error vs CASCI
  - CX/two-qubit gates
  - total gates
  - sequence composition: noops/singles/doubles or Pauli-token weights
- [ ] Compare directly against Table I H4 GQKAE `100.0 ± 3.7` two-qubit and `314.0 ± 15.0` total gates.

## Verification

- Unit tests:
  - `uv run pytest -q`
- Smoke commands:
  - `uv run python scripts/train_mvp.py --config configs/h4_gate_match_cited_smoke.yaml`
  - `uv run python scripts/evaluate_mvp.py --run-dir runs/h4_gate_match_cited_smoke`
- Aggregated sweep command, after adding the script:
  - `uv run python scripts/run_h4_gate_match_experiments.py --matrix configs/h4_gate_match_matrix.yaml`
  - `uv run python scripts/summarize_h4_gate_match.py --runs runs/h4_gate_match_*`

## Success criteria

- Primary: best/mean H4 GQKAE circuits land within or very near paper Table I:
  - two-qubit: `100.0 ± 3.7`
  - total: `314.0 ± 15.0`
- Secondary: maintain chemical accuracy:
  - `|E_GQKAE - E_CASCI| <= 0.0016 Ha`
- Tertiary: document any remaining mismatch as a specific modeling/configuration difference, not a gate-counting ambiguity.
