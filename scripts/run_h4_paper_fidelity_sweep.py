#!/usr/bin/env python
"""Run a multi-seed H4 paper-fidelity sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gqkae.config import apply_overrides, load_config
from gqkae.training.runner import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/h4_paper_fidelity.yaml")
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--output-prefix", default="runs/h4_paper_fidelity_seed")
    p.add_argument("--backend", default=None, help="Optional backend override, e.g. determinant or cudaq")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Small CPU transformer/QSCI workloads are substantially faster without
    # oversubscribed BLAS/OpenMP thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    summaries = []
    for seed in args.seeds:
        out_dir = Path(f"{args.output_prefix}{seed}")
        if args.skip_existing and (out_dir / "summary.json").exists():
            summaries.append(json.loads((out_dir / "summary.json").read_text()))
            continue
        overrides = [
            f"experiment.seed={seed}",
            f"experiment.output_dir={out_dir}",
        ]
        if args.backend:
            overrides.append(f"qsci.backend={args.backend}")
        cfg = apply_overrides(load_config(args.config), overrides)
        summaries.append(train(cfg))
    Path("runs").mkdir(exist_ok=True)
    aggregate_path = Path(f"{args.output_prefix}_aggregate.json")
    aggregate_path.write_text(json.dumps({"runs": summaries}, indent=2) + "\n")
    print(json.dumps({"aggregate": str(aggregate_path), "n_runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
