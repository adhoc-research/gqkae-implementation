#!/usr/bin/env python
"""Generate content-equivalent H4 replication plots from available artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(path: Path):
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if rows and "bond_length_angstrom" in rows[0]:
        return rows
    raise ValueError(f"unsupported summary format: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gqkae-summary", default="runs/h4_pes_cudaq_nvidia_summary.json")
    p.add_argument("--output-dir", default="docs/figures")
    args = p.parse_args()

    rows = _load_rows(Path(args.gqkae_summary))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_r = {}
    for r in rows:
        by_r.setdefault(float(r["bond_length_angstrom"]), []).append(r)
    xs = sorted(by_r)
    casci = [sum(float(r["casci_energy"]) for r in by_r[x]) / len(by_r[x]) for x in xs]
    gqkae = [sum(float(r["gqkae_energy"]) for r in by_r[x]) / len(by_r[x]) for x in xs]
    err = [sum(float(r["abs_error_vs_casci"]) for r in by_r[x]) / len(by_r[x]) for x in xs]

    plt.figure()
    plt.plot(xs, casci, marker="o", label="CASCI")
    plt.plot(xs, gqkae, marker="s", label="GQKAE")
    plt.xlabel("H-H distance (Å)")
    plt.ylabel("Energy (Ha)")
    plt.title("H4 potential-energy surface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "h4_pes.png", dpi=200)
    plt.close()

    plt.figure()
    plt.semilogy(xs, err, marker="o", label="GQKAE |E-CASCI|")
    plt.axhline(0.0016, color="red", linestyle="--", label="chemical accuracy")
    plt.xlabel("H-H distance (Å)")
    plt.ylabel("Absolute error (Ha)")
    plt.title("H4 absolute error vs CASCI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "h4_abs_error.png", dpi=200)
    plt.close()

    # Convergence from the first available R=0.88 run history.
    r088 = [r for r in rows if abs(float(r["bond_length_angstrom"]) - 0.88) < 1e-9]
    if r088:
        hist_path = Path(r088[0]["run_dir"]) / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            plt.figure()
            plt.plot([h["iteration"] for h in hist], [h["best_so_far_energy"] for h in hist], label="best so far")
            plt.axhline(float(r088[0]["casci_energy"]), color="black", linestyle="--", label="CASCI")
            plt.xlabel("Iteration")
            plt.ylabel("Energy (Ha)")
            plt.title("H4 convergence at R=0.88 Å")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "h4_convergence_R0p88.png", dpi=200)
            plt.close()

    print(json.dumps({"output_dir": str(out_dir), "plots": ["h4_pes.png", "h4_abs_error.png", "h4_convergence_R0p88.png"]}, indent=2))


if __name__ == "__main__":
    main()
