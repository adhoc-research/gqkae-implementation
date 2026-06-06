# CUDA-Q H4 PES five-trial implementation plan

## Context

The CUDA-Q feasibility benchmark found that exact CUDA-Q H4 PES with five independent trials per grid point appears locally reasonable on the `nvidia` target: conservative ETA about `7.6h`, linear-fit range about `2.1–2.4h`. The user now wants to plan the implementation for this exact CUDA-Q PES run and add runtime logging.

This phase should execute the full H4 PES/absolute-error backend-fidelity evidence needed to remove the CUDA-Q PES caveat, using exact paper settings rather than reduced settings.

## Approach

Add a dedicated CUDA-Q PES multi-trial runner that wraps the existing `train(config)` workflow, but adds robust orchestration, target overrides, resume behavior, and structured runtime logging at the per-seed, per-bond, per-run, and aggregate levels.

Use the GPU CUDA-Q target discovered during feasibility:

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

Run five independent trials for each paper H4 PES bond length:

```text
[0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30] Å
```

Default seeds should be `[0, 1, 2, 3, 4]`, matching the paper's five independent trials protocol.

## Files to modify

- `scripts/run_h4_pes.py`
  - Either extend it for multi-seed CUDA-Q PES or leave it as single-seed and create a new dedicated wrapper.
- New recommended script: `scripts/run_h4_cudaq_pes_five_trials.py`
- `src/gqkae/training/runner.py`
  - Add optional detailed runtime metrics in `history.json`/`summary.json`.
- New config: `configs/h4_paper_fidelity_cudaq_nvidia.yaml`
- Docs:
  - `docs/CUDAQ_H4_FEASIBILITY_REPORT.md`
  - `docs/H4_REPLICATION_STATUS.md`
  - Optional final PES report: `docs/CUDAQ_H4_PES_FIVE_TRIAL_REPORT.md`

## Reuse

Existing reusable pieces:

- `scripts/run_h4_pes.py`: bond-grid orchestration pattern and summary row schema.
- `scripts/run_h4_paper_fidelity_sweep.py`: multi-seed orchestration and `--skip-existing` behavior.
- `src/gqkae/training/runner.py`: exact training loop and per-iteration `history.json`/`best.json`/`summary.json` artifacts.
- `configs/h4_paper_fidelity.yaml`: exact paper settings except target needs `nvidia` instead of `qpp-cpu`.
- `scripts/cudaq_target_diagnostics.py`: can be reused at run start to record CUDA-Q/GPU environment.

## Runtime logging requirements

Add structured timing at these levels:

1. **Aggregate run level**
   - wall-clock start/end timestamps
   - total elapsed seconds
   - CUDA-Q target/option
   - seeds, grid, shots, iterations, batch size
   - number of completed/skipped/failed jobs

2. **Per bond × seed job level**
   - bond length
   - seed
   - output directory
   - start/end timestamps
   - elapsed seconds
   - status: `completed`, `skipped_existing`, or `failed`
   - exception string/traceback for failures
   - best energy/error and chemical-accuracy flag if completed

3. **Per training iteration level**
   - iteration elapsed seconds
   - sequence sampling/generation time
   - circuit evaluation total time across batch
   - policy update/training time
   - optional mean per-circuit evaluation time

4. **Optional per-circuit evaluation level**
   - only if log size remains reasonable or behind `--detailed-timing`
   - CUDA-Q sample/evaluate elapsed time per generated sequence
   - QSCI elapsed time per sequence

Write these artifacts:

- Aggregate JSON: `runs/h4_pes_cudaq_nvidia_five_trials_summary.json`
- Aggregate runtime JSONL: `runs/h4_pes_cudaq_nvidia_five_trials_runtime.jsonl`
- Aggregate markdown report: `runs/h4_pes_cudaq_nvidia_five_trials_report.md`
- Existing per-run artifacts remain under directories like:
  - `runs/h4_pes_cudaq_nvidia_R0p88_seed0/summary.json`
  - `runs/h4_pes_cudaq_nvidia_R0p88_seed0/history.json`
  - `runs/h4_pes_cudaq_nvidia_R0p88_seed0/best.json`

## Steps

- [ ] Step 1: Create `configs/h4_paper_fidelity_cudaq_nvidia.yaml` from the approved paper-fidelity config, changing only CUDA-Q target fields and output naming defaults.
- [ ] Step 2: Add timing instrumentation to `src/gqkae/training/runner.py` so each iteration records generation, evaluation, optimization, and total elapsed time without changing training behavior.
- [ ] Step 3: Add optional per-circuit evaluation timing in the runner, controlled by a lightweight flag or always summarized as aggregate evaluation time to avoid bloated logs.
- [ ] Step 4: Implement `scripts/run_h4_cudaq_pes_five_trials.py` with CLI options for config, seeds, bond grid, target, option, output prefix, skip-existing, stop-on-failure, and detailed timing.
- [ ] Step 5: In the PES runner, call CUDA-Q diagnostics once at start and store the result in the aggregate artifact.
- [ ] Step 6: In the PES runner, execute the Cartesian product of 10 bond lengths × 5 seeds = 50 exact jobs, with resumable per-job output directories and robust failure capture.
- [ ] Step 7: Add aggregate summarization: mean/std energy, absolute error, chemical-accuracy pass rate, mean/std gate counts, runtime statistics per bond and overall.
- [ ] Step 8: Generate a markdown report with PES table, absolute-error table, runtime table, failures/skips, and exact command/config used.
- [ ] Step 9: Run a one-job smoke test using exact CUDA-Q target but reduced iterations/shots via CLI overrides or a smoke config to validate logging and resume behavior.
- [ ] Step 10: Run the full exact CUDA-Q PES five-trial job on `nvidia` after smoke passes.
- [ ] Step 11: Update `docs/H4_REPLICATION_STATUS.md` and add `docs/CUDAQ_H4_PES_FIVE_TRIAL_REPORT.md` with final backend-fidelity results.

## Verification

- `uv run --extra paper pytest -q` passes.
- CUDA-Q diagnostics confirm `nvidia` target is usable before the run starts.
- Smoke run writes all expected timing artifacts and can be resumed with `--skip-existing`.
- Full run completes 50/50 bond × seed jobs, or any failures are captured with tracebacks and can be resumed.
- Every completed exact CUDA-Q PES point reports energy, CASCI reference, absolute error, and chemical-accuracy flag.
- Aggregate report includes runtime totals and per-job timing distribution.
- CUDA-Q backend caveat can be removed only after exact CUDA-Q PES evidence and exact CUDA-Q Table-I/convergence evidence are both present.

## Proposed command after implementation

```bash
uv run --extra paper --extra cuda python scripts/run_h4_cudaq_pes_five_trials.py \
  --config configs/h4_paper_fidelity_cudaq_nvidia.yaml \
  --seeds 0 1 2 3 4 \
  --target nvidia \
  --output-prefix runs/h4_pes_cudaq_nvidia \
  --skip-existing
```
