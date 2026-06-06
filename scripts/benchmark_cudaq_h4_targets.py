#!/usr/bin/env python
"""Benchmark fixed-sequence H4 paper-fidelity CUDA-Q CPU/GPU targets.

This is a feasibility benchmark, not a final acceptance run. It times identical fixed
H4 0.88 A paper-pool circuits at multiple shot counts and projects the cost of exact
paper Table-I and PES CUDA-Q jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from gqkae.chemistry import build_h4_problem
from gqkae.circuits import simulate_full_qubit_state_vector
from gqkae.config import apply_overrides, load_config
from gqkae.cudaq_backend import (
    _sample_result_to_counts,
    build_cudaq_kernel,
    configure_cudaq_target,
    estimate_cudaq_resources,
)
from gqkae.operator_pool import build_operator_pool
from gqkae.qsci import qsci_energy_from_counts

from cudaq_target_diagnostics import probe_targets

DEFAULT_SEQUENCE = [6, 5, 0, 12, 9, 1, 20, 10, 20, 20, 1, 2, 16, 15, 10, 14, 5, 14, 14, 3]
DEFAULT_PES_GRID = [0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 1.00, 1.10, 1.20, 1.30]


def _import_cudaq():
    import cudaq  # type: ignore

    return cudaq


def _target_label(target: str | None, option: str | None) -> str:
    return f"{target or 'default'}" + (f":{option}" if option else "")


def _normal_approx_count_tolerance(expected_p: float, shots: int, z: float = 5.0) -> float:
    return z * math.sqrt(max(expected_p * (1.0 - expected_p), 1e-12) / float(shots)) + 1.0 / shots


def _index_to_qubit_bitstring(index: int, n_qubits: int) -> str:
    return "".join("1" if ((index >> i) & 1) else "0" for i in range(n_qubits))


def _deterministic_probabilities(sequence, pool, basis) -> dict[str, float]:
    """Full-qubit deterministic measurement probabilities for CUDA-Q equivalence.

    Do not use the determinant-sector simulator here: paper Pauli-word circuits can place
    small probability mass outside the fixed-N/Sz determinant basis, and QSCI filters it
    after sampling. CUDA-Q measures the full qubit register, so the correct equivalence
    reference is the full-qubit state vector.
    """
    state = simulate_full_qubit_state_vector(sequence, pool, basis)
    probs = np.abs(state) ** 2
    return {
        _index_to_qubit_bitstring(i, basis.n_qubits): float(p)
        for i, p in enumerate(probs)
        if p > 1e-15
    }


def _distribution_metrics(sample_probs: dict[str, float], ref_probs: dict[str, float], shots: int) -> dict[str, Any]:
    keys = sorted(set(sample_probs) | set(ref_probs))
    l1 = sum(abs(sample_probs.get(k, 0.0) - ref_probs.get(k, 0.0)) for k in keys)
    max_abs = max((abs(sample_probs.get(k, 0.0) - ref_probs.get(k, 0.0)) for k in keys), default=0.0)
    violations = []
    for k in keys:
        p = ref_probs.get(k, 0.0)
        delta = abs(sample_probs.get(k, 0.0) - p)
        tol = _normal_approx_count_tolerance(p, shots)
        if delta > tol:
            violations.append({"bitstring": k, "sample_p": sample_probs.get(k, 0.0), "ref_p": p, "delta": delta, "tol_5sigma": tol})
    return {
        "l1_distance": float(l1),
        "max_abs_probability_delta": float(max_abs),
        "n_5sigma_binomial_violations": len(violations),
        "violations_preview": violations[:10],
    }


def _load_sequence(path: str | None) -> list[int]:
    if not path:
        return list(DEFAULT_SEQUENCE)
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [int(x) for x in data]
    if "sequence" in data:
        return [int(x) for x in data["sequence"]]
    if "best" in data and "sequence" in data["best"]:
        return [int(x) for x in data["best"]["sequence"]]
    raise ValueError(f"could not find sequence in {path}")


def _fit_eta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("success")]
    if not usable:
        return {"error": "no successful timing rows"}
    shots = np.array([float(r["shots"]) for r in usable])
    per_sequence = np.array([float(r["timing_s"]["total"]) for r in usable])
    sample_only = np.array([float(r["timing_s"]["sample"]) for r in usable])

    # Conservative observed-rate scaling from largest shot count.
    max_row = max(usable, key=lambda r: int(r["shots"]))
    max_shots = float(max_row["shots"])
    max_total = float(max_row["timing_s"]["total"])
    max_sample = float(max_row["timing_s"]["sample"])
    fixed_non_sample = max(0.0, max_total - max_sample)
    rate_scaled_100k = fixed_non_sample + max_sample * (100000.0 / max_shots)

    fit_payload: dict[str, Any] = {
        "observed_max_shot_count": int(max_shots),
        "rate_scaled_from_largest_shots": {"one_sequence_100000_shots_s": rate_scaled_100k},
    }
    if len(usable) >= 2:
        total_coeff = np.polyfit(shots, per_sequence, deg=1)
        sample_coeff = np.polyfit(shots, sample_only, deg=1)
        total_fit_100k = float(np.polyval(total_coeff, 100000.0))
        sample_fit_100k = float(np.polyval(sample_coeff, 100000.0))
        fit_payload["linear_fit_total_vs_shots"] = {
            "slope_s_per_shot": float(total_coeff[0]),
            "intercept_s": float(total_coeff[1]),
            "one_sequence_100000_shots_s": max(total_fit_100k, 0.0),
        }
        fit_payload["linear_fit_sample_vs_shots_plus_fixed"] = {
            "sample_slope_s_per_shot": float(sample_coeff[0]),
            "sample_intercept_s": float(sample_coeff[1]),
            "one_sequence_100000_shots_s": max(sample_fit_100k + fixed_non_sample, 0.0),
        }

    assumptions = []
    for name, one_seq_s in [
        ("rate_scaled_from_largest_shots", rate_scaled_100k),
        (
            "linear_fit_total_vs_shots",
            fit_payload.get("linear_fit_total_vs_shots", {}).get("one_sequence_100000_shots_s"),
        ),
        (
            "linear_fit_sample_vs_shots_plus_fixed",
            fit_payload.get("linear_fit_sample_vs_shots_plus_fixed", {}).get("one_sequence_100000_shots_s"),
        ),
    ]:
        if one_seq_s is None:
            continue
        one_seq_s = float(one_seq_s)
        one_iter_s = one_seq_s * 10.0
        one_trial_s = one_iter_s * 100.0
        five_trials_s = one_trial_s * 5.0
        pes_one_seed_s = one_trial_s * len(DEFAULT_PES_GRID)
        pes_five_trials_per_grid_s = pes_one_seed_s * 5.0
        assumptions.append(
            {
                "assumption": name,
                "table_i": {
                    "one_sequence_100000_shots_s": one_seq_s,
                    "one_iteration_batch10_s": one_iter_s,
                    "one_trial_100_iterations_s": one_trial_s,
                    "five_independent_trials_s": five_trials_s,
                },
                "pes_projection_only": {
                    "grid_points": len(DEFAULT_PES_GRID),
                    "one_seed_all_grid_points_s": pes_one_seed_s,
                    "five_trials_per_grid_point_s": pes_five_trials_per_grid_s,
                },
            }
        )
    fit_payload["eta_assumptions"] = assumptions
    return fit_payload


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = ["# CUDA-Q H4 local feasibility benchmark", ""]
    lines.append("This is a feasibility benchmark, not a final exact-paper acceptance run.")
    lines.append("")
    lines.append(f"Sequence: `{payload['sequence']}`")
    lines.append(f"Shot counts: `{payload['shot_counts']}`")
    lines.append("")
    lines.append("## Targets")
    for target in payload["targets"]:
        lines.append(f"- `{target['label']}` target=`{target['target']}` option=`{target.get('option')}`")
    lines.append("")
    lines.append("## Timing")
    lines.append("| target | shots | setup | build | sample | resources | QSCI | total | energy err | L1 prob |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["benchmark_rows"]:
        if not row.get("success"):
            lines.append(f"| {row['target_label']} | {row['shots']} | FAIL | | | | | | | |")
            continue
        t = row["timing_s"]
        lines.append(
            f"| {row['target_label']} | {row['shots']} | {t['target_setup']:.3f} | {t['kernel_build']:.3f} | "
            f"{t['sample']:.3f} | {t['resource_estimation']:.3f} | {t['qsci_solve']:.3f} | {t['total']:.3f} | "
            f"{row['qsci']['energy_error_vs_casci']:.3e} | {row['probability_metrics']['l1_distance']:.3f} |"
        )
    lines.append("")
    lines.append("## ETA projections")
    for target_label, eta in payload["eta_by_target"].items():
        lines.append(f"### {target_label}")
        if "error" in eta:
            lines.append(f"- {eta['error']}")
            continue
        for item in eta["eta_assumptions"]:
            table = item["table_i"]
            pes = item["pes_projection_only"]
            lines.append(f"- assumption `{item['assumption']}`:")
            lines.append(f"  - one 100000-shot sequence: {_fmt_seconds(table['one_sequence_100000_shots_s'])}")
            lines.append(f"  - one exact iteration, batch 10: {_fmt_seconds(table['one_iteration_batch10_s'])}")
            lines.append(f"  - one exact 100-iteration trial: {_fmt_seconds(table['one_trial_100_iterations_s'])}")
            lines.append(f"  - five exact trials: {_fmt_seconds(table['five_independent_trials_s'])}")
            lines.append(f"  - PES one seed over grid: {_fmt_seconds(pes['one_seed_all_grid_points_s'])} (projection only)")
            lines.append(f"  - PES five trials/grid point: {_fmt_seconds(pes['five_trials_per_grid_point_s'])} (projection only)")
    lines.append("")
    lines.append("Exact five-trial and PES runs were **not run** in this phase and need user greenlight.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/h4_paper_fidelity.yaml")
    p.add_argument("--shot-counts", nargs="*", type=int, default=[100, 1000, 5000, 10000])
    p.add_argument("--sequence-json", default=None, help="Optional JSON with sequence or summary/best.sequence")
    p.add_argument("--output-json", default="runs/cudaq_h4_target_benchmark.json")
    p.add_argument("--output-md", default="runs/cudaq_h4_target_benchmark.md")
    p.add_argument("--include-failed-targets", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    # The benchmark sets shots directly per row; keep the exact paper method settings otherwise.
    cfg.qsci.backend = "cudaq"
    sequence = _load_sequence(args.sequence_json)

    diagnostics = probe_targets(shots=100)
    targets = []
    for row in diagnostics.get("candidates", []):
        if not row.get("usable"):
            if args.include_failed_targets:
                targets.append({"label": row["label"], "target": row["target"], "option": row.get("option"), "usable": False, "error": row.get("error")})
            continue
        if row.get("label") == "qpp-cpu" or row.get("label") == diagnostics.get("selected_gpu", {}).get("label"):
            targets.append({"label": row["label"], "target": row["target"], "option": row.get("option"), "usable": True})
    if not any(t["label"] == "qpp-cpu" for t in targets):
        raise RuntimeError("qpp-cpu was not usable; cannot establish CPU baseline")

    problem = build_h4_problem(cfg.molecule)
    pool = build_operator_pool(problem, cfg.operator_pool)
    ref_probs = _deterministic_probabilities(sequence, pool, problem.basis)

    cudaq = _import_cudaq()
    benchmark_rows = []
    for target in targets:
        if not target.get("usable"):
            continue
        for shots in args.shot_counts:
            label = str(target["label"])
            t_total0 = time.perf_counter()
            row: dict[str, Any] = {"target_label": label, "target": target["target"], "option": target.get("option"), "shots": int(shots)}
            try:
                t0 = time.perf_counter()
                configure_cudaq_target(target=str(target["target"]), option=target.get("option"), seed=cfg.experiment.seed)
                t_setup = time.perf_counter() - t0

                t0 = time.perf_counter()
                kernel = build_cudaq_kernel(sequence, pool, problem.basis)
                t_build = time.perf_counter() - t0

                t0 = time.perf_counter()
                sample_result = cudaq.sample(kernel, shots_count=int(shots))
                t_sample = time.perf_counter() - t0
                counts = _sample_result_to_counts(sample_result, problem.n_qubits, cfg.qsci.cudaq_reverse_bitstrings)
                probs = {k: v / float(shots) for k, v in counts.items()}

                t0 = time.perf_counter()
                resources = estimate_cudaq_resources(sequence, pool, problem.basis)
                t_resources = time.perf_counter() - t0

                t0 = time.perf_counter()
                qsci = qsci_energy_from_counts(
                    counts,
                    problem,
                    dmax=cfg.qsci.dmax,
                    add_hf_det=cfg.qsci.add_hf_det,
                    enlarge_method=cfg.qsci.enlarge_method,
                )
                t_qsci = time.perf_counter() - t0

                row.update(
                    {
                        "success": True,
                        "timing_s": {
                            "target_setup": t_setup,
                            "kernel_build": t_build,
                            "sample": t_sample,
                            "resource_estimation": t_resources,
                            "qsci_solve": t_qsci,
                            "total": time.perf_counter() - t_total0,
                        },
                        "counts_n_bitstrings": len(counts),
                        "top_counts": sorted(counts.items(), key=lambda kv: -kv[1])[:10],
                        "probability_metrics": _distribution_metrics(probs, ref_probs, int(shots)),
                        "qsci": {
                            "energy": qsci.energy,
                            "energy_error_vs_casci": qsci.energy - problem.casci_energy,
                            "subspace_dimension": qsci.subspace_dimension,
                        },
                        "resources": resources,
                    }
                )
            except Exception as exc:
                row.update({"success": False, "error": str(exc), "timing_s": {"total": time.perf_counter() - t_total0}})
            benchmark_rows.append(row)
            print(json.dumps({"target": label, "shots": shots, "success": row.get("success"), "total_s": row.get("timing_s", {}).get("total")}), flush=True)

    eta_by_target = {}
    for target in targets:
        if not target.get("usable"):
            continue
        label = str(target["label"])
        eta_by_target[label] = _fit_eta([r for r in benchmark_rows if r.get("target_label") == label])

    payload = {
        "phase": "cudaq-local-feasibility-not-final-acceptance",
        "config": args.config,
        "sequence": sequence,
        "shot_counts": [int(x) for x in args.shot_counts],
        "diagnostics": diagnostics,
        "targets": targets,
        "references": {
            "casci_energy": problem.casci_energy,
            "hf_bitstring": problem.basis.hf_bitstring,
            "n_qubits": problem.n_qubits,
            "n_determinants": problem.n_determinants,
            "pes_grid_for_eta": DEFAULT_PES_GRID,
        },
        "benchmark_rows": benchmark_rows,
        "eta_by_target": eta_by_target,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown_report(payload))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
