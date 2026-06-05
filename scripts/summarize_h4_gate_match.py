"""Summarize H4 gate-match run directories."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, stdev


def _summary(path: Path) -> dict | None:
    p = path / "summary.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    best = data.get("best", {})
    gate = best.get("gate_count", {}) or {}
    return {
        "run_dir": str(path),
        "seed": data.get("config", {}).get("experiment", {}).get("seed"),
        "spec": data.get("operator_pool", {}).get("spec"),
        "vocab_size": data.get("operator_pool", {}).get("vocab_size"),
        "energy_error": best.get("error"),
        "two_qubit": gate.get("two_qubit"),
        "total": gate.get("total"),
        "composition": best.get("sequence_composition", {}),
    }


def _stats(rows: list[dict], key: str) -> dict[str, float] | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs or glob patterns")
    parser.add_argument("--output", default="runs/h4_gate_match_summary.json")
    args = parser.parse_args()
    dirs: list[Path] = []
    for pat in args.runs:
        matches = glob.glob(pat)
        dirs.extend(Path(m) for m in (matches or [pat]))
    rows = [r for d in dirs if (r := _summary(d)) is not None]
    aggregate = {
        "paper_table_i_h4_gqkae": {"two_qubit_mean": 100.0, "two_qubit_std": 3.7, "total_mean": 314.0, "total_std": 15.0},
        "n_runs": len(rows),
        "energy_error": _stats(rows, "energy_error"),
        "two_qubit": _stats(rows, "two_qubit"),
        "total": _stats(rows, "total"),
        "runs": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
