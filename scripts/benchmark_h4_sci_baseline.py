#!/usr/bin/env python
"""Feasibility benchmark for an H4 selected-CI/SCI-style baseline.

For H4 CAS(4,4), the fixed-Sz CASCI space has only 36 determinants, so selected-CI
is expected to collapse to or closely match full active-space CASCI very quickly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gqkae.chemistry import build_h4_problem
from gqkae.config import MoleculeConfig

DEFAULT_GRID = [0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bond-grid", nargs="*", type=float, default=DEFAULT_GRID)
    p.add_argument("--output", default="runs/h4_sci_feasibility.json")
    args = p.parse_args()

    rows = []
    start = time.perf_counter()
    for bond in args.bond_grid:
        row_start = time.perf_counter()
        cfg = MoleculeConfig(bond_length_angstrom=float(bond), spin_orbital_ordering="interleaved")
        problem = build_h4_problem(cfg)
        # Exact diagonalization of the 36-determinant active-space Hamiltonian is the
        # selected-CI endpoint for this tiny H4 active space. Also report how many
        # determinants would be available to any SCI selector.
        eig_start = time.perf_counter()
        sci_energy = float(np.linalg.eigvalsh(problem.hamiltonian)[0])
        eig_s = time.perf_counter() - eig_start
        rows.append({
            "bond_length_angstrom": float(bond),
            "n_determinants_full_active_space": int(problem.n_determinants),
            "hf_energy": float(problem.hf_energy),
            "ccsd_energy": float(problem.ccsd_energy),
            "casci_energy": float(problem.casci_energy),
            "sci_endpoint_energy": sci_energy,
            "abs_error_vs_casci": abs(sci_energy - float(problem.casci_energy)),
            "build_and_solve_elapsed_s": time.perf_counter() - row_start,
            "active_hamiltonian_eig_elapsed_s": eig_s,
        })
    payload = {
        "method": "H4 CAS(4,4) selected-CI endpoint / full active-space diagonalization",
        "note": "The active space has only 36 fixed-Sz determinants, so SCI feasibility is effectively the CASCI endpoint and is locally cheap.",
        "grid": [float(x) for x in args.bond_grid],
        "elapsed_s": time.perf_counter() - start,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
