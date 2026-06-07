#!/usr/bin/env python
"""Generate content-equivalent H4 replication markdown tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean(vals):
    return statistics.fmean(vals) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gqkae-summary", default="runs/h4_pes_cudaq_nvidia_summary.json")
    p.add_argument("--output-dir", default="docs/tables")
    args = p.parse_args()

    payload = json.loads(Path(args.gqkae_summary).read_text())
    rows = payload["rows"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_r = {}
    for r in rows:
        by_r.setdefault(float(r["bond_length_angstrom"]), []).append(r)

    energy_lines = ["# H4 PES/error table", "", "| R (Å) | CASCI (Ha) | GQKAE mean (Ha) | mean |error| (Ha) | seeds |", "|---:|---:|---:|---:|---:|"]
    for r in sorted(by_r):
        group = by_r[r]
        energy_lines.append(
            f"| {r:.2f} | {mean([float(x['casci_energy']) for x in group]):.12f} | "
            f"{mean([float(x['gqkae_energy']) for x in group]):.12f} | "
            f"{mean([float(x['abs_error_vs_casci']) for x in group]):.3e} | {len(group)} |"
        )
    (out / "h4_pes_error_table.md").write_text("\n".join(energy_lines) + "\n")

    r088 = by_r.get(0.88, [])
    gate_lines = ["# H4 Table-I-style gate counts", "", "| Method | two-qubit mean | total excl. HF-X mean | source |", "|---|---:|---:|---|"]
    if r088:
        gate_lines.append(
            f"| GQKAE local | {mean([float(x['two_qubit']) for x in r088]):.1f} | "
            f"{mean([float(x['total_excluding_hf_x']) for x in r088]):.1f} | existing CUDA-Q PES |"
        )
    gate_lines.append("| GQE local | TBD | TBD | pending paper-scale production |")
    gate_lines.append("| VQE local | TBD | TBD | pending baseline feasibility/production |")
    (out / "h4_gate_count_table.md").write_text("\n".join(gate_lines) + "\n")

    model_lines = ["# H4 Table-II-style model/runtime table", "", "| Method | parameters | parameter memory MB | wall time | source |", "|---|---:|---:|---:|---|"]
    model_lines.append("| GQKAE MVP local | see run summaries | see run summaries | existing PES mean ~261.9s/job | existing CUDA-Q PES |")
    model_lines.append("| GQKAE paper-scale local | TBD | TBD | TBD | feasibility pending |")
    model_lines.append("| GQE paper-scale local | TBD | TBD | TBD | feasibility pending |")
    (out / "h4_model_runtime_table.md").write_text("\n".join(model_lines) + "\n")

    print(json.dumps({"output_dir": str(out), "tables": ["h4_pes_error_table.md", "h4_gate_count_table.md", "h4_model_runtime_table.md"]}, indent=2))


if __name__ == "__main__":
    main()
