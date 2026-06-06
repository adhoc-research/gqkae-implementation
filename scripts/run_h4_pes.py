#!/usr/bin/env python
"""Run H4 paper-fidelity PES / absolute-error sweep over a bond grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gqkae.config import apply_overrides, load_config
from gqkae.training.runner import train

DEFAULT_H4_PAPER_GRID = [0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/h4_paper_fidelity.yaml")
    p.add_argument("--bond-grid", nargs="*", type=float, default=DEFAULT_H4_PAPER_GRID)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--backend", default="determinant")
    p.add_argument("--output-prefix", default="runs/h4_pes_paper_fidelity")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rows = []
    for bond in args.bond_grid:
        bond_label = f"{bond:.2f}".replace(".", "p")
        out_dir = Path(f"{args.output_prefix}_R{bond_label}_seed{args.seed}")
        if args.skip_existing and (out_dir / "summary.json").exists():
            summary = json.loads((out_dir / "summary.json").read_text())
        else:
            overrides = [
                f"molecule.bond_length_angstrom={bond}",
                f"experiment.seed={args.seed}",
                f"experiment.output_dir={out_dir}",
                f"qsci.backend={args.backend}",
            ]
            cfg = apply_overrides(load_config(args.config), overrides)
            summary = train(cfg)
        best = summary["best"]
        refs = summary["references"]
        rows.append({
            "bond_length_angstrom": float(bond),
            "run_dir": str(out_dir),
            "seed": int(args.seed),
            "backend": summary["config"]["qsci"]["backend"],
            "hf_energy": float(refs["hf_energy"]),
            "ccsd_energy": float(refs.get("ccsd_energy", float("nan"))),
            "casci_energy": float(refs["casci_energy"]),
            "gqkae_energy": float(best["energy"]),
            "abs_error_vs_casci": abs(float(best["error"])),
            "two_qubit": float(best.get("paper_table_i_count_view", {}).get("two_qubit", best["gate_count"].get("two_qubit", 0))),
            "total_excluding_hf_x": float(best.get("paper_table_i_count_view", {}).get("total_excluding_hf_x", best["gate_count"].get("total", 0) - best["gate_count"].get("x", 0))),
        })
    payload = {
        "grid": [float(x) for x in args.bond_grid],
        "seed": int(args.seed),
        "chemical_accuracy_ha": 0.0016,
        "all_chemical_accuracy": all(r["abs_error_vs_casci"] <= 0.0016 for r in rows),
        "rows": rows,
    }
    out = Path(f"{args.output_prefix}_summary_seed{args.seed}.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(out), "all_chemical_accuracy": payload["all_chemical_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
