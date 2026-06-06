#!/usr/bin/env python
"""Run exact CUDA-Q H4 PES over the paper grid with five independent trials."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gqkae.config import apply_overrides, as_dict, load_config
from gqkae.training.runner import train

from cudaq_target_diagnostics import probe_targets

DEFAULT_H4_PAPER_GRID = [0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
CHEMICAL_ACCURACY_HA = 0.0016


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bond_label(bond: float) -> str:
    return f"{bond:.2f}".replace(".", "p")


def _stats(values: list[float]) -> dict[str, float | None]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _paper_count_view(best: dict[str, Any]) -> dict[str, float]:
    view = best.get("paper_table_i_count_view") or {}
    gate_count = best.get("gate_count") or {}
    return {
        "two_qubit": float(view.get("two_qubit", gate_count.get("two_qubit", 0.0)) or 0.0),
        "total_excluding_hf_x": float(
            view.get("total_excluding_hf_x", (gate_count.get("total", 0.0) or 0.0) - (gate_count.get("x", 0.0) or 0.0))
        ),
        "total_including_hf_x": float(view.get("total_including_hf_x", gate_count.get("total", 0.0)) or 0.0),
    }


def _row_from_summary(summary: dict[str, Any], bond: float, seed: int, out_dir: Path, status: str, elapsed_s: float | None) -> dict[str, Any]:
    best = summary["best"]
    refs = summary["references"]
    timing = summary.get("timing_s", {})
    counts = _paper_count_view(best)
    abs_error = abs(float(best["error"]))
    return {
        "status": status,
        "bond_length_angstrom": float(bond),
        "seed": int(seed),
        "run_dir": str(out_dir),
        "backend": summary["config"]["qsci"]["backend"],
        "cudaq_target": summary["config"]["qsci"].get("cudaq_target"),
        "cudaq_option": summary["config"]["qsci"].get("cudaq_option"),
        "shots": int(summary["config"]["qsci"]["shots"]),
        "iterations": int(summary["config"]["training"]["iterations"]),
        "batch_circuits": int(summary["config"]["training"]["batch_circuits"]),
        "hf_energy": float(refs["hf_energy"]),
        "ccsd_energy": float(refs.get("ccsd_energy", float("nan"))),
        "casci_energy": float(refs["casci_energy"]),
        "gqkae_energy": float(best["energy"]),
        "error_vs_casci": float(best["error"]),
        "abs_error_vs_casci": abs_error,
        "chemical_accuracy": bool(abs_error <= CHEMICAL_ACCURACY_HA),
        "best_iteration": int(best.get("iteration", -1)),
        "two_qubit": counts["two_qubit"],
        "total_excluding_hf_x": counts["total_excluding_hf_x"],
        "total_including_hf_x": counts["total_including_hf_x"],
        "elapsed_s": float(elapsed_s if elapsed_s is not None else timing.get("training_total", 0.0) or 0.0),
        "train_timing_s": timing,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") in {"completed", "skipped_existing"} and "gqkae_energy" in r]
    failed = [r for r in rows if r.get("status") == "failed"]
    by_bond = []
    for bond in sorted({float(r["bond_length_angstrom"]) for r in completed}):
        br = [r for r in completed if float(r["bond_length_angstrom"]) == bond]
        by_bond.append(
            {
                "bond_length_angstrom": bond,
                "n_completed": len(br),
                "energy": _stats([r["gqkae_energy"] for r in br]),
                "abs_error_vs_casci": _stats([r["abs_error_vs_casci"] for r in br]),
                "chemical_accuracy_pass_rate": float(sum(1 for r in br if r["chemical_accuracy"]) / len(br)) if br else 0.0,
                "two_qubit": _stats([r["two_qubit"] for r in br]),
                "total_excluding_hf_x": _stats([r["total_excluding_hf_x"] for r in br]),
                "elapsed_s": _stats([r["elapsed_s"] for r in br]),
                "casci_energy": float(br[0]["casci_energy"]) if br else None,
            }
        )
    return {
        "n_jobs": len(rows),
        "n_completed_or_skipped": len(completed),
        "n_failed": len(failed),
        "all_completed": len(failed) == 0 and len(completed) == len(rows),
        "all_chemical_accuracy": bool(completed) and all(r["chemical_accuracy"] for r in completed),
        "overall": {
            "energy": _stats([r["gqkae_energy"] for r in completed]),
            "abs_error_vs_casci": _stats([r["abs_error_vs_casci"] for r in completed]),
            "chemical_accuracy_pass_rate": float(sum(1 for r in completed if r["chemical_accuracy"]) / len(completed)) if completed else 0.0,
            "two_qubit": _stats([r["two_qubit"] for r in completed]),
            "total_excluding_hf_x": _stats([r["total_excluding_hf_x"] for r in completed]),
            "elapsed_s": _stats([r["elapsed_s"] for r in completed]),
        },
        "by_bond": by_bond,
        "failures": failed,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = ["# CUDA-Q H4 PES five-trial report", ""]
    lines.append("Exact CUDA-Q PES run over the H4 paper grid with five independent trials per bond length.")
    lines.append("")
    lines.append(f"- target: `{payload['cudaq_target']}` option=`{payload.get('cudaq_option')}`")
    lines.append(f"- shots: `{payload['config']['qsci']['shots']}`")
    lines.append(f"- iterations: `{payload['config']['training']['iterations']}`")
    lines.append(f"- batch circuits: `{payload['config']['training']['batch_circuits']}`")
    lines.append(f"- total elapsed: `{payload['elapsed_s']:.1f}s`")
    lines.append(f"- completed/skipped jobs: `{summary['n_completed_or_skipped']}/{summary['n_jobs']}`")
    lines.append(f"- failures: `{summary['n_failed']}`")
    lines.append(f"- all chemical accuracy: `{summary['all_chemical_accuracy']}`")
    lines.append("")
    lines.append("## PES energy table")
    lines.append("| R (Å) | n | CASCI (Ha) | mean GQKAE (Ha) | std GQKAE | mean abs error (Ha) | chem pass | mean CX | mean total excl. HF x | mean runtime |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["by_bond"]:
        lines.append(
            f"| {row['bond_length_angstrom']:.2f} | {row['n_completed']} | {row['casci_energy']:.12f} | "
            f"{row['energy']['mean']:.12f} | {row['energy']['std']:.3e} | {row['abs_error_vs_casci']['mean']:.3e} | "
            f"{row['chemical_accuracy_pass_rate']:.2f} | {row['two_qubit']['mean']:.1f} | "
            f"{row['total_excluding_hf_x']['mean']:.1f} | {row['elapsed_s']['mean']:.1f}s |"
        )
    lines.append("")
    lines.append("## Per-job rows")
    lines.append("| R (Å) | seed | status | energy | abs error | CX | total excl. HF x | elapsed | run dir |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        if row.get("status") == "failed":
            lines.append(f"| {row['bond_length_angstrom']:.2f} | {row['seed']} | failed | | | | | {row.get('elapsed_s', 0):.1f}s | `{row['run_dir']}` |")
            continue
        lines.append(
            f"| {row['bond_length_angstrom']:.2f} | {row['seed']} | {row['status']} | {row['gqkae_energy']:.12f} | "
            f"{row['abs_error_vs_casci']:.3e} | {row['two_qubit']:.1f} | {row['total_excluding_hf_x']:.1f} | "
            f"{row['elapsed_s']:.1f}s | `{row['run_dir']}` |"
        )
    if summary["failures"]:
        lines.append("")
        lines.append("## Failures")
        for failure in summary["failures"]:
            lines.append(f"- R={failure['bond_length_angstrom']}, seed={failure['seed']}: `{failure.get('error')}`")
    lines.append("")
    lines.append("## Command")
    lines.append("```bash")
    lines.append("uv run --extra paper --extra cuda python " + " ".join(payload.get("argv", [])))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/h4_paper_fidelity_cudaq_nvidia.yaml")
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--bond-grid", nargs="*", type=float, default=DEFAULT_H4_PAPER_GRID)
    p.add_argument("--target", default="nvidia")
    p.add_argument("--option", default=None)
    p.add_argument("--output-prefix", default="runs/h4_pes_cudaq_nvidia")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--detailed-timing", action="store_true")
    p.add_argument("--override", action="append", default=[], help="Extra config override section.field=value; useful for smoke tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.detailed_timing:
        os.environ["GQKAE_DETAILED_TIMING"] = "1"

    aggregate_start = time.perf_counter()
    start_timestamp = _utc_now()
    diagnostics = probe_targets(shots=100)
    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg.qsci.backend = "cudaq"
    base_cfg.qsci.cudaq_target = args.target
    base_cfg.qsci.cudaq_option = args.option

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{args.output_prefix}_summary.json")
    runtime_jsonl = Path(f"{args.output_prefix}_runtime.jsonl")
    report_path = Path(f"{args.output_prefix}_report.md")
    runtime_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with runtime_jsonl.open("a", encoding="utf-8") as log:
        for bond in args.bond_grid:
            for seed in args.seeds:
                out_dir = Path(f"{args.output_prefix}_R{_bond_label(bond)}_seed{seed}")
                job_start = time.perf_counter()
                job_record: dict[str, Any] = {
                    "event": "job_start",
                    "timestamp": _utc_now(),
                    "bond_length_angstrom": float(bond),
                    "seed": int(seed),
                    "run_dir": str(out_dir),
                }
                log.write(json.dumps(job_record) + "\n")
                log.flush()
                try:
                    if args.skip_existing and (out_dir / "summary.json").exists():
                        summary = json.loads((out_dir / "summary.json").read_text())
                        elapsed_s = float(summary.get("timing_s", {}).get("training_total", 0.0) or 0.0)
                        row = _row_from_summary(summary, bond, seed, out_dir, "skipped_existing", elapsed_s)
                    else:
                        cfg = apply_overrides(load_config(args.config), args.override)
                        cfg.qsci.backend = "cudaq"
                        cfg.qsci.cudaq_target = args.target
                        cfg.qsci.cudaq_option = args.option
                        cfg.molecule.bond_length_angstrom = float(bond)
                        cfg.experiment.seed = int(seed)
                        cfg.experiment.output_dir = str(out_dir)
                        summary = train(cfg)
                        elapsed_s = time.perf_counter() - job_start
                        row = _row_from_summary(summary, bond, seed, out_dir, "completed", elapsed_s)
                    rows.append(row)
                    log.write(json.dumps({"event": "job_end", "timestamp": _utc_now(), **row}) + "\n")
                    log.flush()
                    print(json.dumps({"bond": bond, "seed": seed, "status": row["status"], "elapsed_s": row["elapsed_s"], "abs_error": row["abs_error_vs_casci"]}), flush=True)
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "bond_length_angstrom": float(bond),
                        "seed": int(seed),
                        "run_dir": str(out_dir),
                        "elapsed_s": float(time.perf_counter() - job_start),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    rows.append(row)
                    log.write(json.dumps({"event": "job_failed", "timestamp": _utc_now(), **row}) + "\n")
                    log.flush()
                    print(json.dumps({"bond": bond, "seed": seed, "status": "failed", "error": str(exc)}), flush=True)
                    if args.stop_on_failure:
                        break
            if args.stop_on_failure and rows and rows[-1].get("status") == "failed":
                break

    elapsed_s = time.perf_counter() - aggregate_start
    payload = {
        "phase": "exact-cudaq-h4-pes-five-trials",
        "start_timestamp": start_timestamp,
        "end_timestamp": _utc_now(),
        "elapsed_s": float(elapsed_s),
        "argv": sys.argv,
        "config_path": args.config,
        "config": as_dict(base_cfg),
        "cudaq_target": args.target,
        "cudaq_option": args.option,
        "diagnostics": diagnostics,
        "grid": [float(x) for x in args.bond_grid],
        "seeds": [int(x) for x in args.seeds],
        "chemical_accuracy_ha": CHEMICAL_ACCURACY_HA,
        "rows": rows,
        "summary": _summarize_rows(rows),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(report_path, payload)
    print(json.dumps({"summary": str(summary_path), "report": str(report_path), "elapsed_s": elapsed_s, "completed": payload["summary"]["n_completed_or_skipped"], "failed": payload["summary"]["n_failed"]}, indent=2))


if __name__ == "__main__":
    main()
