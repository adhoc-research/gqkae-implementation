# CUDA-Q paper-backend match plan

## Context

The H4 paper-fidelity work currently has a strong algorithmic replication claim: Tequila-backed cited operator-pool construction, CCSD coefficients, interleaved ordering, QSCI symmetry completion, and paper-style gate counting. The remaining caveat is backend fidelity: the accepted 10-seed/PES artifacts were produced with the deterministic/statevector sampler, while the paper used CUDA-Q simulation.

This phase is a **local feasibility exploration** for CUDA-Q backend matching. Before committing to long exact-paper CUDA-Q jobs, we will benchmark local CUDA-Q CPU vs GPU execution and estimate runtime at exact settings. The output of this phase is a report for user review, not a final acceptance run and not an automatic backend/config decision.

## Approach

Use CUDA-Q benchmark and validation runs to evaluate local feasibility, while preserving deterministic runs as fast regression references. The plan will verify bitstring conventions, circuit/source equivalence, resource counting, stochastic reproducibility, and runtime feasibility. It will stop at a recommendation/report; exact five-trial acceptance jobs are deferred until the user greenlights a specific target/config in a later phase.

User decisions for this CUDA-Q phase:

- First run a **smoke ablation** comparing CUDA-Q `qpp-cpu` vs CUDA-Q GPU acceleration speed on the same H4 paper-fidelity workload.
- The machine has one GPU; the diagnostic/benchmark script should discover CUDA-Q target availability and auto-select the GPU target if possible rather than requiring the user to know the target name up front.
- Benchmark multiple shot counts up to `10000` shots for CPU and GPU, then report estimated ETA for exact paper Table-I settings and H4 PES settings.
- Keep the eventual paper-fidelity target in mind as **five independent CUDA-Q trials**, matching the paper's stated averaging protocol.
- For this feasibility experiment, do **not** run or claim final acceptance. Instead, estimate the cost of eventual exact-paper settings: `shots=100000`, `iterations=100`, `batch_circuits=10`, sequence length `L=20`, QSCI `dmax=2000`, and paper-fidelity operator pool.
- Produce enough timing/equivalence evidence for the user to choose the next-phase CUDA-Q configuration.

Important implementation detail from the current code: `src/gqkae/cudaq_backend.py` already builds CUDA-Q kernels through `cudaq.make_kernel()`, applies HF `x` gates, emits paper-style JW Pauli exponentials, samples with `cudaq.sample`, and attaches `cudaq.estimate_resources` metadata. `src/gqkae/circuits.py` dispatches `backend="cudaq"` through `_sample_with_cudaq_or_raise`. Therefore the plan should harden and validate the existing path rather than replace it.

## Files to modify

Initial likely files:

- `src/gqkae/cudaq_backend.py`
- `src/gqkae/circuits.py`
- `src/gqkae/training/runner.py`
- `src/gqkae/config.py`
- `configs/h4_paper_fidelity.yaml`
- New CUDA-Q-specific configs/scripts under `configs/` and `scripts/`
- `pyproject.toml` only if CUDA-Q optional dependency pins/target extras need adjustment
- New tests under `tests/`
- Final docs under `docs/`

## Reuse

Existing code/patterns to reuse:

- CUDA-Q kernel builder and sampler in `src/gqkae/cudaq_backend.py`.
- Backend dispatch in `src/gqkae/circuits.py`.
- Training loop metadata/history/summary output in `src/gqkae/training/runner.py`.
- Existing paper-fidelity config settings in `configs/h4_paper_fidelity.yaml`, which already defaults to `qsci.backend=cudaq`, `cudaq_target=qpp-cpu`, `shots=100000`, `iterations=100`, and `batch_circuits=10`.
- Existing multi-seed script `scripts/run_h4_paper_fidelity_sweep.py`; it already accepts `--backend cudaq`, `--skip-existing`, and per-seed output isolation.
- Existing PES script `scripts/run_h4_pes.py`; it supports backend override but defaults to determinant, so CUDA-Q PES should use `--backend cudaq` or a CUDA-Q-specific config/script wrapper.

## Steps

- [ ] Step 1: Add a CUDA-Q target diagnostics script that reports CUDA-Q version, visible GPU information where available, random-seed support, resource-estimator behavior, and which targets/options appear usable locally.
- [ ] Step 2: Implement GPU target auto-discovery for the single-GPU local machine. Try known CUDA-Q target names/options in a safe smoke order, record failures, and select the first working GPU-accelerated target; always include `qpp-cpu` as the CPU baseline.
- [ ] Step 3: Add a fixed-sequence CUDA-Q benchmark script for H4 0.88 Å using the paper-fidelity pool. Benchmark identical sequence/circuit workloads on CPU and GPU targets at multiple shot counts up to `10000` shots, for example `[100, 1000, 5000, 10000]`.
- [ ] Step 4: The benchmark script should report wall-clock timing split into target setup, kernel build, sampling, resource estimation, QSCI solve, and total time. It should write JSON and a readable markdown/console summary.
- [ ] Step 5: Estimate ETA for exact paper Table-I settings from the benchmark: one sequence at `100000` shots, one iteration (`batch_circuits=10`), one full trial (`100` iterations), and five independent trials. Report estimates for every CPU/GPU target and shot-count fit/scaling assumption.
- [ ] Step 5b: Estimate ETA for H4 PES/absolute-error CUDA-Q work from the same benchmark: one exact run per extracted H4 bond length, total grid cost, and variants for one seed vs five independent trials per grid point. Mark PES estimates as projections only, not runs.
- [ ] Step 6: Audit current CUDA-Q sampling path against deterministic/statevector path for fixed H4 sequences.
- [ ] Step 7: Add fixed-sequence CUDA-Q equivalence checks: probabilities/counts within sampling error, bitstring orientation, QSCI subspace, energy, and gate resources.
- [ ] Step 8: Write a feasibility report summarizing CPU/GPU target availability, timing, numerical consistency, Table-I ETA projections, and PES ETA projections. Include recommended next-phase configs, but do **not** automatically choose or run final acceptance jobs.
- [ ] Step 9: Draft next-phase commands/config snippets for likely exact-paper runs, including five independent Table-I trials and exact CUDA-Q PES, clearly marked as **not run in this phase** and awaiting user greenlight.
- [ ] Step 10: Update replication/status docs only to record feasibility findings and explicitly state that exact CUDA-Q acceptance remains pending until the user selects a config.

## Verification

- Unit tests pass with CUDA-Q unavailable by skipping CUDA-Q-only tests where appropriate.
- CUDA-Q diagnostics identify `qpp-cpu` and either auto-select a working GPU target or record why GPU target discovery failed.
- CPU/GPU benchmark artifacts include shot-count scaling up to `10000` shots and ETA projections for exact paper settings.
- Fixed-sequence CUDA-Q sampled energies agree with deterministic probabilities within finite-shot error.
- Fixed-sequence CUDA-Q gate counts match paper/count convention and deterministic static counts.
- Feasibility report gives ETA projections for eventual exact five-trial H4 Table-I runs and CUDA-Q H4 PES runs, but does not claim those runs were completed.
- Final docs for this phase distinguish between algorithmic replication, qpp-cpu feasibility, GPU feasibility, fixed-sequence validation, and deferred exact five-trial/PES acceptance evidence. They must state that the CUDA-Q backend caveat is **not removed** until exact CUDA-Q PES is also run; this phase only estimates whether that is locally feasible.

## Clarifications

### Why not make 10 seeds the main CUDA-Q claim?

The earlier deterministic acceptance artifact used 10 seeds to reduce uncertainty and make our comparison robust. The paper itself reports averages over **five independent trials**. Therefore the most faithful CUDA-Q Table-I reproduction is five exact-paper trials, not 10. A 10-seed CUDA-Q run is useful extra evidence and better statistics, but it is less directly aligned with the paper's stated averaging protocol.

### What is fixed-sequence CUDA-Q validation?

Fixed-sequence validation means: take an already chosen operator-token sequence, build the CUDA-Q circuit for that exact sequence, sample it with CUDA-Q, and compare the resulting bitstrings/QSCI energy/gate resources to deterministic or known-reference behavior. This isolates backend correctness from model training randomness. It answers: "Does CUDA-Q execute and sample our paper-fidelity circuit correctly?" It does **not** by itself prove the full training loop reproduces the paper.

### What is a CUDA-Q PES sweep?

PES means potential-energy surface. For H4, the paper plots energy/error over a grid of bond lengths. A CUDA-Q PES sweep means running the paper-fidelity workflow with `backend=cudaq` at each extracted H4 bond length and reporting GQKAE energy and absolute error vs CASCI. This is stronger than only validating the 0.88 Å Table-I point, but it is much more expensive because it multiplies the exact-paper training workload by every bond length.

### What is most faithful to the paper?

Most faithful priority order:

1. Exact CUDA-Q backend execution, preferably GPU if available and stable.
2. Exact paper H4 Table-I settings at 0.88 Å.
3. Five independent trials, matching the paper's averaging protocol.
4. Per-iteration convergence/history aggregated over those trials.
5. H4 PES/absolute-error over the paper grid, ideally with CUDA-Q and exact settings, if runtime permits.

## Open questions

- Which CUDA-Q GPU target names/options are supported by the installed CUDA-Q version? This will be discovered by the diagnostics script.
- None for this phase. Exact CUDA-Q PES is required to fully remove the backend caveat, but this feasibility phase only estimates its ETA so the user can decide whether/when to run it.
