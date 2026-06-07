#!/usr/bin/env python3
"""Build the static project-page dataset from run artifacts.

Reads the committed ``runs/*.json`` summaries and a handful of per-run
``history.json`` trajectories, then emits ``assets/data.js`` as a single inline
``window.GQKAE_DATA = {...}`` blob. Keeping the data inline (rather than fetched)
makes the page self-contained: it renders from ``file://`` and from GitHub Pages
without a server or CORS headaches.

Usage:
    python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "assets" / "data.js"

# Per-run trajectories to embed for the interactive convergence panel. Kept small
# on purpose so assets/data.js stays light.
CONVERGENCE_RUNS = [
    ("R0.88 · seed0", "h4_pes_cudaq_nvidia_R0p88_seed0"),
    ("R0.50 · seed0", "h4_pes_cudaq_nvidia_R0p50_seed0"),
    ("R1.30 · seed0", "h4_pes_cudaq_nvidia_R1p30_seed0"),
]


def load_json(path: Path):
    if not path.exists():
        print(f"  ! missing: {path.relative_to(REPO_ROOT)}")
        return None
    with path.open() as fh:
        return json.load(fh)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def build_pes(summary):
    """Aggregate the 50-job PES run into per-bond rows + per-seed points."""
    rows = summary.get("rows", [])
    grid = summary.get("grid", [])
    by_bond: dict[float, list[dict]] = {}
    for r in rows:
        by_bond.setdefault(r["bond_length_angstrom"], []).append(r)

    bonds = []
    points = []  # flat per-seed records for scatter panels
    for r_val in sorted(by_bond):
        group = by_bond[r_val]
        casci = group[0]["casci_energy"]
        hf = group[0].get("hf_energy")
        ccsd = group[0].get("ccsd_energy")
        bonds.append(
            {
                "R": r_val,
                "casci": casci,
                "hf": hf,
                "ccsd": ccsd,
                "gqkae": mean_std([g["gqkae_energy"] for g in group]),
                "abs_error": mean_std([g["abs_error_vs_casci"] for g in group]),
                "two_qubit": mean_std([g["two_qubit"] for g in group]),
                "total": mean_std([g["total_excluding_hf_x"] for g in group]),
                "runtime_s": mean_std([g["elapsed_s"] for g in group]),
                "n": len(group),
            }
        )
        for g in group:
            points.append(
                {
                    "R": r_val,
                    "seed": g["seed"],
                    "gqkae": g["gqkae_energy"],
                    "abs_error": g["abs_error_vs_casci"],
                    "two_qubit": g["two_qubit"],
                    "total": g["total_excluding_hf_x"],
                    "runtime_s": g["elapsed_s"],
                    "best_iteration": g.get("best_iteration"),
                    "chemical_accuracy": g.get("chemical_accuracy"),
                }
            )

    seeds = summary.get("seeds", sorted({p["seed"] for p in points}))
    n_jobs = len(rows)
    n_pass = sum(1 for r in rows if r.get("chemical_accuracy"))
    overall = {
        "n_jobs": n_jobs,
        "n_pass": n_pass,
        "n_fail": sum(1 for r in rows if r.get("status") != "completed"),
        "total_runtime_s": summary.get("elapsed_s"),
        "mean_abs_error": statistics.fmean([r["abs_error_vs_casci"] for r in rows]) if rows else None,
        "mean_two_qubit": statistics.fmean([r["two_qubit"] for r in rows]) if rows else None,
        "mean_total": statistics.fmean([r["total_excluding_hf_x"] for r in rows]) if rows else None,
        "chem_accuracy_ha": summary.get("chemical_accuracy_ha", 0.0016),
        "shots": summary.get("config", {}).get("qsci", {}).get("shots"),
        "iterations": summary.get("config", {}).get("training", {}).get("iterations"),
        "batch_circuits": summary.get("config", {}).get("training", {}).get("batch_circuits"),
    }
    return {"grid": grid, "seeds": seeds, "bonds": bonds, "points": points, "overall": overall}


def build_convergence():
    runs = {}
    for label, dirname in CONVERGENCE_RUNS:
        hist = load_json(RUNS / dirname / "history.json")
        if not hist:
            continue
        runs[label] = {
            "iteration": [h["iteration"] for h in hist],
            "best_error": [abs(h.get("best_so_far_error", 0.0)) for h in hist],
            "mean_energy": [h.get("mean_energy") for h in hist],
            "best_energy": [h.get("best_so_far_energy") for h in hist],
            "mean_loss": [h.get("mean_loss") for h in hist],
            "subspace_dim": [h.get("mean_subspace_dimension") for h in hist],
            "two_qubit": [h.get("mean_two_qubit_gates") for h in hist],
        }
    return runs


def build_models():
    families = [
        ("MVP (compact)", "h4_mvp_gqkae_model_scale_benchmark.json", "gqkae"),
        ("GQKAE (paper-scale)", "h4_paper_scale_gqkae_model_scale_benchmark.json", "gqkae"),
        ("GQE (paper-scale)", "h4_paper_scale_gqe_model_scale_benchmark.json", "gqe"),
    ]
    out = []
    for label, fname, family in families:
        d = load_json(RUNS / fname)
        if not d:
            continue
        timing = d.get("timing_s", {})
        out.append(
            {
                "label": label,
                "family": family,
                "parameters": d.get("parameters"),
                "param_memory_mb": d.get("parameter_memory_mb"),
                "rss_peak_mb": d.get("rss_peak_mb"),
                "opt_update_mean_s": timing.get("opt_update_mean"),
                "forward_batch_s": timing.get("forward_batch"),
                "sample_batch_s": timing.get("sample_batch_mean"),
            }
        )
    return out


def build_runtime_breakdown(summary):
    """Mean per-iteration timing split across PES jobs."""
    rows = summary.get("rows", [])
    keys = ("generation_total", "evaluation_total", "optimization_total")
    means = {}
    for k in keys:
        per_iter = []
        for r in rows:
            t = r.get("train_timing_s", {}).get(k, {})
            if t.get("mean_s") is not None:
                per_iter.append(t["mean_s"])
        means[k.replace("_total", "")] = statistics.fmean(per_iter) if per_iter else None
    return means


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def main():
    print("Building site data from runs/ ...")
    pes_summary = load_json(RUNS / "h4_pes_cudaq_nvidia_summary.json")
    fidelity = load_json(RUNS / "h4_paper_fidelity_sweep_summary.json") or {}

    if pes_summary is None:
        raise SystemExit("Cannot build site: runs/h4_pes_cudaq_nvidia_summary.json is required.")

    pes = build_pes(pes_summary)
    convergence = build_convergence()
    models = build_models()
    runtime = build_runtime_breakdown(pes_summary)

    diag = pes_summary.get("diagnostics", {})
    gpu = (diag.get("nvidia_smi", {}).get("gpus") or [{}])[0]
    paper_targets = fidelity.get("paper_table_i_h4_gqkae", {
        "two_qubit_mean": 100.0, "two_qubit_std": 3.7, "total_mean": 314.0, "total_std": 15.0,
    })

    data = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": git_commit(),
            "source": "runs/h4_pes_cudaq_nvidia_summary.json",
            "environment": {
                "gpu": gpu.get("name"),
                "gpu_memory_mib": gpu.get("memory_total_mib"),
                "cudaq_version": diag.get("cudaq_version"),
                "python": diag.get("platform", {}).get("python"),
                "platform": diag.get("platform", {}).get("platform"),
            },
        },
        "molecule": pes_summary.get("config", {}).get("molecule", {}),
        "paper_targets": paper_targets,
        "pes": pes,
        "convergence": convergence,
        "models": models,
        "runtime": runtime,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False)
    OUT.write_text(
        "// Auto-generated by scripts/build_site_data.py - do not edit by hand.\n"
        f"window.GQKAE_DATA = {payload};\n"
    )

    print(f"  wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  PES: {len(pes['bonds'])} bond lengths, {len(pes['points'])} per-seed jobs")
    print(f"  convergence runs: {len(convergence)}")
    print(f"  model-scale entries: {len(models)}")
    print(f"  overall: {pes['overall']['n_pass']}/{pes['overall']['n_jobs']} chemical-accuracy pass")


if __name__ == "__main__":
    main()
