# GQKAE — H4

A paper-faithful implementation of the **Generative Quantum-inspired
Kolmogorov-Arnold Eigensolver** (GQKAE) for the H4 `(4e, 4o)` / 6-31G / `L=20` benchmark.
The full paper is included in the repo root
(`Generative Quantum-inspired Kolmogorov-Arnold Eigensolver.pdf`).

The implementation runs the paper's full training loop:

1. **PySCF** builds the H4 active space (CAS(4,4)) reference and CASCI ground truth.
2. An **UCCSD-derived discrete operator pool** defines a vocabulary of excitation tokens.
3. An autoregressive **HQKANsformer policy** generates length-`L` operator sequences.
4. Each sequence becomes a state-preparation circuit, sampled either by a real **CUDA-Q**
   backend or a **determinant-space state-vector** fallback.
5. **QSCI** (quantum-selected configuration interaction) diagonalizes the sampled
   subspace to produce an energy estimate / reward.
6. **GRPO** updates the policy from sequence-level QSCI rewards.

**Accuracy target:** `|E_GQKAE − E_CASCI| ≤ 0.0016 Ha` (chemical accuracy).

## Repository layout

```
gqkae/                     core package
  chemistry.py             PySCF H4 active space, determinant basis, CASCI reference
  operator_pool.py         UCCSD-derived discrete excitation vocabulary
  fermion_mapping.py       Jordan-Wigner fermion→Pauli transform
  circuits.py              sequence→circuit + determinant-space simulator
  cudaq_backend.py         CUDA-Q sampling and resource estimation
  gate_counting.py         paper-style all-to-all Pauli-evolution gate counts
  qsci.py                  quantum-selected CI subspace diagonalization
  reporting.py             run evaluation / report formatting
  models/
    hqkansformer.py        autoregressive HQKANsformer policy
    qkan.py                quantum-inspired KAN / DARUAN activations
    gpt_policy.py          GPT/GQE-style baseline policy
  training/
    runner.py              end-to-end training loop (entry point)
    grpo.py                clipped GRPO objective and advantages
configs/                   YAML experiment configs (smoke → paper-fidelity)
data/                      cited Pauli-evolution operator-pool reference
scripts/                   train / evaluate / benchmark / PES / reporting scripts
tests/                     pytest suite (11 modules)
docs/                      feasibility + replication-status reports, figures, tables
runs/                      committed run artifacts (used by the dashboard)
index.html, assets/        static results dashboard (GitHub Pages)
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for package management; the
in-repo `.venv` holds the Python interpreter.

```bash
uv venv
uv sync --extra dev

# Optional on supported Linux systems for a real quantum backend:
uv sync --extra dev --extra cuda

# Optional: Tequila-based cited operator-pool construction:
uv sync --extra dev --extra paper
```

## Quick start (determinant backend)

The determinant-space simulator runs anywhere (no CUDA-Q needed) and is used for tests
and local smoke runs:

```bash
uv run python scripts/train_mvp.py \
  --config configs/h4_mvp.yaml \
  --override training.iterations=2 \
  --override qsci.shots=200 \
  --override qsci.dmax=20

uv run python scripts/evaluate_mvp.py --run-dir runs/h4_mvp
```

Configs default to small "smoke" values; inline comments in `configs/h4_mvp.yaml` show
the paper-like settings (e.g. `shots=100000`, `dmax=2000`, `iterations=100`,
`batch_circuits=10`). Override any field with `--override key=value`.

## CUDA-Q runs

```bash
# Low-shot CPU-simulator smoke (requires `uv sync --extra dev --extra cuda`):
uv run python scripts/train_mvp.py --config configs/h4_cudaq_smoke.yaml
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_cudaq_smoke

# Paper-like H4 settings; set `qsci.cudaq_target` to `nvidia` / `tensornet` as appropriate:
uv run python scripts/train_mvp.py --config configs/h4_cudaq_paper_like.yaml
```

## Potential energy surface

```bash
# Full H4 PES grid with multiple trials per bond length:
uv run python scripts/run_h4_pes.py
uv run python scripts/run_h4_cudaq_pes_five_trials.py
```

## Gate counting

Resource reporting uses the paper-style **all-to-all Pauli-evolution** convention:
arbitrary `rz` rotations, `cx` ladders, and `h` / `s` / `sdg` Clifford basis changes,
with HF-reference `x` gates reported separately. This matches the cited `gqe-for-qsci`
decomposition (see `gqkae/gate_counting.py`).

## Results

The exact CUDA-Q `nvidia` backend completed the full paper H4 PES grid with five
independent trials per bond length under exact paper settings (`shots=100000`,
`iterations=100`, `L=20`, QSCI `dmax=2000` with symmetry completion):

- **50/50 jobs at chemical accuracy** across the bond-length grid.
- At the Table-I bond length `0.88 Å`: mean GQKAE energy `−2.195180549992 Ha` vs CASCI
  `−2.195180549992 Ha` (MAE `≈3.5e-15 Ha`).
- Mean CX/two-qubit gates `100.4` (paper `100.0 ± 3.7`); mean total gates `322.2`
  (paper `314.0 ± 15.0`).

Details: `docs/H4_REPLICATION_STATUS.md`, `docs/CUDAQ_H4_PES_FIVE_TRIAL_REPORT.md`,
and aggregate artifacts under `runs/h4_pes_cudaq_nvidia_*`.

## Tests

```bash
uv run pytest
```

Coverage includes the fermion mapping, gate counting, QSCI (and symmetry completion),
the CUDA-Q backend, the QKAN / HQKANsformer model variants, GRPO, config handling,
spin-orbital ordering, and operator-pool / circuit construction.

## Results dashboard (GitHub Pages)

A self-contained, Grafana-style dashboard lives at `index.html` (+ `assets/`). It
visualizes the committed H4 PES results — energy curve, accuracy, gate budgets,
convergence, and model scaling — directly from `runs/*.json`.

```bash
# regenerate assets/data.js from the run artifacts
uv run python scripts/build_site_data.py

# preview locally (or just open index.html directly)
python -m http.server 8000   # then visit http://localhost:8000/
```

To publish: push to GitHub and set **Settings → Pages → Build from branch → `/ (root)`**.
The page is pure static HTML/CSS/JS (Apache ECharts via CDN) — no build step.
