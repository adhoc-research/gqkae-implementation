# Reimplementation Plan: Generative Quantum-inspired Kolmogorov-Arnold Eigensolver (GQKAE)

## Context

- Repository currently contains the paper PDF (`Generative Quantum-inspired Kolmogorov-Arnold Eigensolver.pdf`) and `AGENTS.md`; there is no implementation code yet.
- The PDF proposes GQKAE: a Generative Quantum Eigensolver (GQE) variant that replaces the GPT-2 feed-forward layers with an HQKAN/QKAN latent processor, then evaluates generated circuits through QSCI and trains the policy with GRPO.
- Desired scope is a minimal MVP, not a full reproduction of all six benchmark systems. The MVP target is H4 with active space `(4e,4o)`, 6-31G basis, generated circuit sequence length `L=20`, and default bond length `0.88 Å` for the paper’s equilibrium-style convergence setting.
- The MVP should still follow the paper’s stack/fidelity choices where practical: `uv`, PySCF, CUDA-Q simulation, QSCI evaluation, GRPO training, and an HQKANsformer generator.
- Intended outcome: a runnable end-to-end training/evaluation path for H4 that demonstrates the full GQKAE loop and can later be expanded to the full paper benchmark suite.

## Approach

Recommended implementation structure:

1. Build a Python project managed with `uv` per `AGENTS.md`.
2. Implement a narrow vertical slice of the paper method for H4 `(4e,4o)`, 6-31G, `L=20`, default bond length `0.88 Å`:
   - H4 molecular problem setup and active-space Hamiltonian generation.
   - UCCSD-derived operator pool and autoregressive operator-token sequences.
   - HQKANsformer policy: replace each transformer block FFN with encoder -> QKAN/DARUAN latent processor -> decoder.
   - CUDA-Q-backed circuit/state simulation, bitstring sampling, QSCI determinant subspace construction, and projected Hamiltonian diagonalization.
   - GRPO training loop over sampled operator sequences using QSCI energy rewards.
3. Include a small GPT-style GQE baseline only if needed to verify the GQKAE parameter reduction and compare behavior; otherwise keep the MVP focused on GQKAE.
4. Keep default experiment settings much smaller than the paper for local smoke tests, while providing config fields for paper-like settings (`M=10`, `N_shots=1e5`, `N_iter=100`, `dmax=2000`, AdamW `lr=5e-6`, weight decay `0.01`, repetition penalty `1.2`).

## Files to modify

No source files exist yet. Proposed MVP files to create during implementation:

- `pyproject.toml` — project metadata and paper-fidelity dependencies.
- `src/gqkae/` — package root.
- `src/gqkae/config.py` — experiment and model configuration dataclasses.
- `src/gqkae/chemistry.py` — H4 setup, `(4e,4o)` active-space integrals, CASCI/reference energy.
- `src/gqkae/operator_pool.py` — UCCSD excitation/operator vocabulary construction.
- `src/gqkae/circuits.py` — sequence-to-CUDA-Q-circuit/state-preparation utilities.
- `src/gqkae/qsci.py` — sampled determinant selection, optional symmetry completion, projected Hamiltonian diagonalization.
- `src/gqkae/models/qkan.py` — DARUAN, QKAN layer, HQKAN module.
- `src/gqkae/models/hqkansformer.py` — decoder-only transformer blocks with HQKAN FFN replacement.
- `src/gqkae/training/grpo.py` — token-wise clipped GRPO loss and advantage normalization.
- `src/gqkae/training/runner.py` — end-to-end MVP training loop.
- `configs/h4_mvp.yaml` — single default H4 `(4e,4o)`, 6-31G, bond length `0.88 Å`, `L=20` MVP experiment config.
- `scripts/train_mvp.py` — run the selected-molecule MVP.
- `scripts/evaluate_mvp.py` — compute best energy/error and basic parameter/gate counts.
- `tests/` — targeted unit and smoke tests.

## Reuse

Existing repository reuse is limited because only the paper PDF is present.

Implementation should reuse external/open-source components where practical:

- PySCF for H4 molecular integrals and CASCI reference energy in the `(4e,4o)` active space with 6-31G.
- PyTorch for the HQKANsformer model and GRPO optimization.
- CUDA-Q for paper-fidelity circuit construction/simulation and measurement sampling.
- SciPy/NumPy for projected Hamiltonian construction and diagonalization.
- Existing QKAN/FlashQKAN ideas from the paper references should guide the DARUAN/QKAN module design, but the MVP can implement the smallest compatible PyTorch version first unless an existing dependency is selected during implementation.

## Steps

- [x] Set up Python packaging and a clear experiment configuration system using `uv`.
- [x] Implement chemistry/reference-energy pipeline for H4 `(4e,4o)` with 6-31G and default bond length `0.88 Å` only.
- [x] Implement UCCSD-derived operator vocabulary and sequence-to-CUDA-Q-circuit conversion for `L=20` H4 circuits.
- [x] Implement QSCI post-processing and validate it on the selected active space.
- [x] Implement DARUAN, QKAN, HQKAN, and HQKANsformer policy modules.
- [x] Implement autoregressive sequence sampling, QSCI reward computation, and GRPO updates.
- [x] Add MVP CLI scripts/config for training and evaluation.
- [x] Add minimal reporting: best-so-far energy error, final CASCI error, parameter count/memory estimate, and approximate gate count.
- [x] Leave full PES scans, all-molecule benchmarks, and publication-style figure reproduction as future expansion work outside this MVP.

## Verification

- Unit tests:
  - QKAN/DARUAN output shapes and gradients.
  - Autoregressive masking and log-probability calculations.
  - GRPO loss on synthetic rewards.
  - QSCI diagonalization on tiny Hamiltonians.
- Smoke tests:
  - Run H4 with very low shots/iterations and small `dmax`.
  - Verify training loop produces finite rewards/losses and writes checkpoints/results.
- Scientific checks:
  - Compare H4 HF/CASCI reference energy against PySCF.
  - Confirm best-so-far QSCI energy error trends downward on the H4 benchmark.
  - Report whether chemical accuracy is reached when using paper-like settings, if hardware/runtime permits.
- Paper-fidelity checks retained for the MVP:
  - Config supports `M=10`, `N_shots=1e5`, `N_iter=100`, `dmax=2000`, GRPO + AdamW (`lr=5e-6`, weight decay `0.01`), repetition penalty `1.2`.
  - CUDA-Q is the primary simulation/sampling backend.

