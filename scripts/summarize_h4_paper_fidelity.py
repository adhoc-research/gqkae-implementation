#!/usr/bin/env python
"""Summarize H4 paper-fidelity runs against the paper target."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _load_summary(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text())


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def summarize(paths: list[str]) -> dict:
    runs = [_load_summary(p) for p in paths]
    rows = []
    for s in runs:
        best = s["best"]
        view = best.get("paper_table_i_count_view") or {}
        gate = best.get("gate_count", {})
        cfg = s.get("config", {})
        rows.append({
            "run_dir": cfg.get("experiment", {}).get("output_dir"),
            "seed": cfg.get("experiment", {}).get("seed"),
            "backend": cfg.get("qsci", {}).get("backend"),
            "pool_sha256": s.get("metadata", {}).get("pool_sha256"),
            "energy_error": float(best.get("error")),
            "two_qubit": float(view.get("two_qubit", gate.get("two_qubit", 0))),
            "total_excluding_hf_x": float(view.get("total_excluding_hf_x", gate.get("total", 0) - gate.get("x", 0))),
            "total_including_hf_x": float(view.get("total_including_hf_x", gate.get("total", 0))),
        })
    out = {
        "paper_table_i_h4_gqkae": {"two_qubit_mean": 100.0, "two_qubit_std": 3.7, "total_mean": 314.0, "total_std": 15.0},
        "n_runs": len(rows),
        "energy_error": _stats([r["energy_error"] for r in rows]),
        "two_qubit": _stats([r["two_qubit"] for r in rows]),
        "total_excluding_hf_x": _stats([r["total_excluding_hf_x"] for r in rows]),
        "total_including_hf_x": _stats([r["total_including_hf_x"] for r in rows]),
        "all_chemical_accuracy": all(abs(r["energy_error"]) <= 0.0016 for r in rows),
        "runs": rows,
    }
    twoq_ok = abs(out["two_qubit"]["mean"] - 100.0) <= max(2 * 3.7, 0.10 * 100.0)
    total_ok = abs(out["total_excluding_hf_x"]["mean"] - 314.0) <= max(2 * 15.0, 0.10 * 314.0)
    out["practical_numerical_agreement"] = bool(twoq_ok and total_ok and out["all_chemical_accuracy"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Run dirs, summary files, or glob patterns")
    p.add_argument("--output", default="runs/h4_paper_fidelity_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[str] = []
    for item in args.runs:
        expanded = sorted(glob.glob(item))
        paths.extend(expanded or [item])
    summary = summarize(paths)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
