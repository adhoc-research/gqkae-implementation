# Plan: Exact paper-style decomposition and gate counting

## Context

- The repository is now a git repository on `master` with the H4 GQKAE MVP committed.
- The user wants the exact paper-style decomposition/counting work done on a new branch, and wants regenerated run artifacts committed too.
- The paper states that Table I gate counts are evaluated assuming all-to-all connectivity and “a standard decomposition into arbitrary-angle single-qubit rotation gates, CX gates, and single-qubit Clifford gates.” For H4 `(4e,4o)` at `0.88 Å`, the reported GQKAE count is approximately `100.0 ± 3.7` two-qubit gates and `314.0 ± 15.0` total gates.
- The paper cites `moken20/gqe-for-qsci`; its `get_pauli_evolution_gate_count(pauli)` counts each Pauli evolution term as:
  - `cx = 0 if weight <= 1 else 2 * (weight - 1)`
  - `h = 2 * (nX + nY)`
  - `s = nY`, `sdg = nY`
  - `rz = 1 if weight >= 1 else 0`
  - `total = cx + h + s + sdg + rz`
- Current implementation already has a real CUDA-Q backend in `src/gqkae/cudaq_backend.py` that emits Jordan-Wigner Pauli rotations for each UCCSD excitation token and uses `cudaq.estimate_resources` when CUDA-Q is available, but its Y-basis change uses `rx(±pi/2)` rather than the paper/cited-repo `sdg`/`h` and `h`/`s` convention.
- Current determinant-backend reporting still uses rough estimates from `src/gqkae/operator_pool.py` (`approximate_gate_count*`).
- `src/gqkae/circuits.py::sequence_to_cudaq_source` explicitly documents that its generated CUDA-Q source uses approximate compact blocks and is the next production step for exact fermionic-excitation decomposition.

## Approach

- Create a feature branch from `master` before implementation.
- Make the paper/cited-repo Pauli-evolution decomposition the single source of truth for counting: arbitrary `rz`, `cx`, `h`, `s`, and `sdg` with all-to-all CNOT ladders.
- Update source generation and CUDA-Q emission to use the same paper-style basis-change convention where supported: X uses `h`, Y uses `sdg` then `h` before the CNOT ladder, and undo uses `h` then `s`.
- Keep CUDA-Q sampling behavior scientifically consistent with the existing backend, but make counting/reporting deterministic and available for both CUDA-Q and determinant fallback runs.
- Add tests that validate decomposition/counting on noop, single, and double excitation sequences without requiring a GPU; CUDA-Q-specific resource tests remain optional.

## Files to modify

- `src/gqkae/cudaq_backend.py`
- `src/gqkae/circuits.py`
- `src/gqkae/operator_pool.py`
- `src/gqkae/reporting.py`
- `tests/test_cudaq_backend.py`
- `tests/test_operator_pool_and_circuit_source.py`
- Add a focused new test file, e.g. `tests/test_gate_counting.py`
- `README.md` / `docs/FUTURE_WORK.md` to move exact H4 decomposition/counting out of future work once implemented.
- Regenerated artifacts under `runs/` after the implementation is verified.

## Reuse

- Reuse `src/gqkae/fermion_mapping.py::excitation_pauli_terms` for exact Jordan-Wigner Pauli terms.
- Reuse the cited `moken20/gqe-for-qsci` counting convention from `gqe_qsci/gqe/utils.py::get_pauli_evolution_gate_count` as the paper-style target.
- Reuse `src/gqkae/cudaq_backend.py::apply_pauli_exponential` structure for the Pauli evolution ladder, but change Y basis handling from `rx(±pi/2)` to the cited/paper-style Clifford sequence (`sdg+h` before, `h+s` after) so emitted source/counts match the paper convention.
- Reuse `src/gqkae/training/runner.py` existing propagation of `sample.gate_count` into history/best/summary.

## Steps

- [ ] Create and switch to a new branch from `master`, likely `feature/exact-decomposition-counting`.
- [ ] Add a paper-style Pauli-evolution counter equivalent to the cited implementation: per Pauli word, count CNOT ladder, X/Y basis Clifford gates, and one arbitrary `rz`.
- [ ] Add exact sequence-level counting that includes HF reference `x` gates separately, exposes `cx`, `two_qubit`, `h`, `s`, `sdg`, `rz`, `x`, and `total`, and avoids approximate rank-based formulas.
- [ ] Replace `approximate_gate_count` usage in determinant fallback and `sequence_to_cudaq_source` with the paper-style counter.
- [ ] Update CUDA-Q resource normalization to consistently expose paper-style static counts even when CUDA-Q resource APIs use different naming or decompose Clifford gates differently.
- [ ] Update generated CUDA-Q source to mirror the exact Pauli-evolution decomposition instead of compact approximate excitation blocks.
- [ ] Update CUDA-Q kernel emission to prefer paper-style Clifford basis changes (`sdg/h` and `h/s`) while preserving fallback compatibility if a CUDA-Q version lacks `s`/`sdg` builder methods.
- [ ] Remove or deprecate approximate-gate wording in reporting/docs.
- [ ] Add tests for exact counts on representative H4 sequences and source emission.
- [ ] Regenerate the requested run artifacts and include them in the branch commit.
- [ ] Run the test suite.

## Verification

- `uv run pytest -q`
- Manual smoke check for deterministic backend counting:
  - `uv run python scripts/train_mvp.py --config configs/h4_mvp.yaml --override training.iterations=1 --override training.batch_circuits=1 --override qsci.shots=100 --override qsci.dmax=20`
  - Confirm `runs/h4_mvp/summary.json` reports paper-style primitive counts rather than approximate counts.
- Regenerate committed artifacts, likely:
  - `uv run python scripts/train_mvp.py --config configs/h4_cudaq_smoke.yaml`
  - `uv run python scripts/evaluate_mvp.py --run-dir runs/h4_cudaq_smoke`
  - A short/default H4 run if runtime permits.
- Optional full paper-like CUDA-Q artifact regeneration if GPU/runtime is available:
  - `uv run python scripts/train_mvp.py --config configs/h4_cudaq_paper_like.yaml`
  - `uv run python scripts/evaluate_mvp.py --run-dir runs/h4_cudaq_paper_like`
- Compare H4 generated-sequence counts against the paper’s Table I scale (`~100` two-qubit / `~314` total for GQKAE H4) and document any remaining deviation if caused by the MVP’s simplified operator pool/model rather than the decomposition/counting convention.

## Resolved decisions

- “Paper-style” means reproduce the paper’s stated all-to-all, arbitrary-rotation/CX/Clifford decomposition and the cited `gqe-for-qsci` Pauli-evolution counting convention, not CUDA-Q’s arbitrary backend-specific resource decomposition.
- Regenerated artifacts under `runs/` should be committed along with source/tests/docs on the feature branch.
