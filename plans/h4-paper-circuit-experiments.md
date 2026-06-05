# Plan: H4 GQKAE circuit-count matching experiments

## Context

The exact paper-style counter is now implemented, but the current H4 paper-like run matches the CASCI energy while greatly exceeding the paper's reported GQKAE circuit sizes:

| Source | two-qubit gates | total gates |
| --- | ---: | ---: |
| Paper Table I, H4 GQKAE | `100.0 ± 3.7` | `314.0 ± 15.0` |
| Current `runs/h4_cudaq_paper_like` best | `1184` | `2728` |

The likely mismatch is not the counting convention anymore. It is the operator/token convention. The cited `moken20/gqe-for-qsci` implementation uses a `PauliEvolutionPool` with defaults in `configs/default.yaml`:

- `operator_pool.spec: pauli_evolution`
- `ccsd_threshold: 1e-6`
- `remove_z_ladder: true`
- `only_use_first_pauli: true`
- `params: null`, meaning CCSD amplitudes determine gate angles

By contrast, this repo currently samples length-20 sequences from full fermionic excitation tokens. A single excitation expands to 2 Pauli evolutions and a double expands to 8 Pauli evolutions, so counts are naturally ~10x larger.

A quick static H4 estimate using the current UCCSD pool shows why the cited Pauli-evolution pool should get us close:

| Token convention | mean CX/token | estimated CX for L=20 | mean total/token | estimated total for L=20 + HF `x` |
| --- | ---: | ---: | ---: | ---: |
| full excitation, keep Z ladder | 55.38 | 1107.69 | 131.69 | 2637.85 |
| full excitation, remove Z ladder | 34.46 | 689.23 | 110.77 | 2219.38 |
| first Pauli only, keep Z ladder | 7.85 | 156.92 | 17.62 | 356.31 |
| first Pauli only, remove Z ladder | 4.77 | 95.38 | 14.54 | 294.77 |

The last row is very close to the paper's H4 GQKAE count scale.

## Approach

Run controlled experiments that change only one source of mismatch at a time, then promote the closest paper-matching variant into a new config/artifact set.

The recommended implementation target for the first experiment branch is a configurable Pauli-evolution operator pool that can reproduce the cited implementation's `remove_z_ladder + only_use_first_pauli` behavior while preserving the existing full-excitation pool for comparison.

## Files to modify

- `src/gqkae/config.py`
  - Add operator-pool options: `spec`, `ccsd_threshold`, `remove_z_ladder`, `only_use_first_pauli`, `use_ccsd_amplitudes`, and possibly `spin_orbital_ordering`.
- `src/gqkae/operator_pool.py`
  - Add a `PauliEvolutionOperator`/pool path or extend existing operators to represent one Pauli word token directly.
  - Keep the existing full-excitation pool as `spec: excitation`.
- `src/gqkae/fermion_mapping.py`
  - Add utilities to remove Z ladders from Pauli words and deduplicate Pauli terms.
- `src/gqkae/circuits.py`
  - Sample/apply Pauli-evolution tokens directly for source generation and deterministic fallback.
- `src/gqkae/cudaq_backend.py`
  - Emit Pauli-evolution tokens directly for CUDA-Q runs.
- `src/gqkae/gate_counting.py`
  - Count Pauli-evolution pools exactly using the existing paper-style per-word counter.
- `src/gqkae/training/runner.py`
  - Record operator-pool `spec`, vocabulary details, and richer gate-count statistics in summaries.
- `configs/h4_cudaq_paper_like.yaml`
  - Preserve existing config or rename to explicitly indicate full-excitation baseline.
- New configs:
  - `configs/h4_paper_pool_smoke.yaml`
  - `configs/h4_paper_pool_paper_like.yaml`
  - Optional sweep configs under `configs/experiments/`.
- New scripts, if useful:
  - `scripts/summarize_operator_pool.py`
  - `scripts/run_h4_gate_count_sweep.py`
- Tests:
  - `tests/test_gate_counting.py`
  - `tests/test_operator_pool_and_circuit_source.py`
  - New `tests/test_pauli_evolution_pool.py`
- Artifacts:
  - New run directories under `runs/h4_paper_pool_*`.

## Reuse

- `src/gqkae/fermion_mapping.py::excitation_pauli_terms`
  - Source of Jordan-Wigner Pauli terms for current UCCSD excitations.
- `src/gqkae/gate_counting.py::paper_style_pauli_evolution_gate_count`
  - Already matches cited per-Pauli-word gate-count formula.
- `src/gqkae/gate_counting.py::paper_style_sequence_gate_count`
  - Can be generalized to support both excitation tokens and Pauli-evolution tokens.
- `src/gqkae/cudaq_backend.py::apply_pauli_exponential`
  - Already emits the paper-style Clifford/CX/RZ decomposition for one Pauli word.
- `src/gqkae/circuits.py::simulate_state_vector`
  - Existing determinant fallback should stay available for full-excitation baseline; Pauli-evolution tokens may need dense-state or determinant-space support depending on approximation fidelity.
- Cited implementation reference:
  - `gqe_qsci/gqe/operator_pool.py::PauliEvolutionPool`
  - `gqe_qsci/gqe/utils.py::get_pauli_evolution_gate_count`

## Experiment matrix

### Phase 0: static pool diagnostics, no training

Goal: verify that token conventions alone can reproduce the paper's reported gate-count scale.

Run/report for each pool variant:

1. Full excitation, keep Z ladder — current baseline.
2. Full excitation, remove Z ladder.
3. First Pauli only, keep Z ladder.
4. First Pauli only, remove Z ladder — cited default.
5. First Pauli only, remove Z ladder, CCSD amplitude screening at thresholds: `1e-8`, `1e-6`, `1e-5`, `1e-4`.
6. Optional spin-orbital ordering comparison: current alpha-first vs cited interleaved ordering.

Metrics:

- vocabulary size
- per-token min/mean/max `cx` and `total`
- expected L=20 random-sequence `cx` and `total`
- count of identity/noop tokens
- top CCSD-amplitude tokens and their gate counts

Acceptance for moving to training:

- A variant has estimated L=20 counts near Table I: `cx` around `100`, total around `314`.

### Phase 1: short smoke training

Goal: ensure the paper-like low-count pool still produces sensible QSCI energies.

Run variants:

1. `spec=pauli_evolution, only_use_first_pauli=true, remove_z_ladder=true, ccsd_threshold=1e-6`
2. Same but `sequence_length=12`
3. Same but `sequence_length=16`
4. Same but `sequence_length=20`

Suggested command pattern:

```bash
uv run python scripts/train_mvp.py \
  --config configs/h4_paper_pool_smoke.yaml \
  --override operator_pool.sequence_length=<L> \
  --override experiment.output_dir=runs/h4_paper_pool_smoke_L<L>
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_paper_pool_smoke_L<L>
```

Smoke settings:

- `shots=1000` to `5000`
- `dmax=100` to `500`
- `iterations=10` to `20`
- `batch_circuits=4` to `10`

Metrics:

- best energy error
- best-sequence `cx` and `total`
- mean gate counts over sampled circuits
- subspace dimension
- vocabulary size

### Phase 2: paper-like single-seed runs

Goal: compare directly against the H4 Table I scale and chemical accuracy target.

Run the best Phase 1 configuration with:

- `shots=100000`
- `dmax=2000`
- `iterations=100`
- `batch_circuits=10`
- seed `7` first, matching existing paper-like config

Acceptance target:

- `two_qubit` within roughly `100 ± 10`
- `total` within roughly `314 ± 30`
- energy within chemical accuracy: `abs(error) <= 0.0016 Ha`

### Phase 3: multi-seed robustness

Goal: check whether counts and accuracy are stable enough to report mean ± std like the paper.

Run seeds:

- `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`

Metrics to aggregate:

- best energy error mean/std
- best `two_qubit` mean/std
- best `total` mean/std
- final-batch best error mean/std
- percentage of runs at chemical accuracy

Acceptance target:

- gate-count mean/std overlaps the paper H4 GQKAE row
- most seeds reach chemical accuracy

### Phase 4: isolate remaining mismatches

If counts are close but energy is not:

- Try `only_use_first_pauli=false` with a shorter `sequence_length` to keep counts near 314.
- Try multiple candidate Pauli terms per excitation selected by lowest count or largest absolute coefficient instead of first term only.
- Try CCSD-amplitude angles vs learned/fixed angle grid.
- Try `params` angle grids if needed, mirroring the cited implementation's optional `params` expansion.

If energy is close but counts are still high:

- Reduce sequence length.
- Increase noop/identity probability or add generation-time count regularization.
- Add a GRPO reward penalty for gate count, e.g. `reward = -energy - lambda * normalized_gate_count`.
- Enforce a hard generation-time gate budget around `cx <= 110` or `total <= 330`.

## Steps

- [ ] Implement static operator-pool diagnostics script and run Phase 0 on the current branch.
- [ ] Add config fields for operator-pool `spec` and cited Pauli-evolution options.
- [ ] Implement `PauliEvolutionPool` compatible with `remove_z_ladder` and `only_use_first_pauli`.
- [ ] Add Pauli-evolution token application/source/counting paths.
- [ ] Add tests for pool construction, deduplication, source emission, and gate counts.
- [ ] Create smoke and paper-like H4 configs for the cited pool.
- [ ] Run Phase 1 smoke sweep and commit summarized artifacts.
- [ ] Run Phase 2 paper-like single-seed experiment.
- [ ] If promising, run Phase 3 multi-seed aggregation.
- [ ] Document whether the closest variant matches Table I gate counts and chemical accuracy.

## Verification

- Unit tests:

```bash
uv run pytest -q
```

- Static diagnostics:

```bash
uv run python scripts/summarize_operator_pool.py --config configs/h4_paper_pool_smoke.yaml
```

- Smoke training/eval:

```bash
uv run python scripts/train_mvp.py --config configs/h4_paper_pool_smoke.yaml
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_paper_pool_smoke
```

- Paper-like training/eval:

```bash
uv run python scripts/train_mvp.py --config configs/h4_paper_pool_paper_like.yaml
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_paper_pool_paper_like
```

## Branch

Created branch:

```text
experiments/h4-paper-gqkae-counts
```
