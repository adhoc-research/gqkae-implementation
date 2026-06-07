# GQKAE MVP

Minimal H4 `(4e,4o)` / 6-31G / `L=20` reimplementation plan and code scaffold for the paper "Generative Quantum-inspired Kolmogorov-Arnold Eigensolver".

The MVP keeps the paper's core loop: PySCF active-space references, UCCSD-derived discrete operator sequences, a real CUDA-Q circuit/sampling backend with a determinant-space fallback, QSCI subspace diagonalization, HQKANsformer policy, and GRPO updates.

## Setup

```bash
uv venv
uv sync --extra dev
# Optional on supported Linux/CUDA-Q systems:
uv sync --extra dev --extra cuda
```

## Determinant-backend smoke run

```bash
uv run python scripts/train_mvp.py \
  --config configs/h4_mvp.yaml \
  --override training.iterations=2 \
  --override qsci.shots=200 \
  --override qsci.dmax=20
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_mvp
```

## CUDA-Q runs

```bash
# Low-shot CPU simulator smoke, requires `uv sync --extra dev --extra cuda`.
uv run python scripts/train_mvp.py --config configs/h4_cudaq_smoke.yaml
uv run python scripts/evaluate_mvp.py --run-dir runs/h4_cudaq_smoke

# Paper-like H4 settings; set `qsci.cudaq_target` to `nvidia`/`tensornet` as appropriate.
uv run python scripts/train_mvp.py --config configs/h4_cudaq_paper_like.yaml
```


Resource reporting uses the paper-style all-to-all Pauli-evolution count convention: arbitrary `rz` rotations, `cx` ladders, and `h`/`s`/`sdg` Clifford basis changes, with HF-reference `x` gates reported separately.

Chemical accuracy target: `|E_GQKAE - E_CASCI| <= 0.0016 Ha`.

## Project page (GitHub Pages)

A self-contained, Grafana-style dashboard lives at `index.html` (+ `assets/`) in the repo
root. It visualizes the committed H4 PES results — energy curve, accuracy, gate budgets,
convergence, and model scaling — directly from `runs/*.json`.

```bash
# regenerate assets/data.js from the run artifacts
python scripts/build_site_data.py

# preview locally (or just open index.html directly)
python -m http.server 8000   # then visit http://localhost:8000/
```

To publish: push to GitHub and set **Settings → Pages → Build from branch → `/ (root)`**.
The page is pure static HTML/CSS/JS (Apache ECharts via CDN) — no build step.
