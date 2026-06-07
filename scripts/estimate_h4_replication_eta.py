#!/usr/bin/env python
"""Estimate H4 paper-faithful production ETAs from feasibility artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def model_job_estimate(model_bench, existing_eval_per_job_s=227.7785322502529, iterations=100):
    t = model_bench["timing_s"]
    # Use measured sample once per iteration and 30-update optimization loop once per iteration.
    gen = float(t["sample_batch_mean"]) * iterations
    opt = float(t["opt_loop_total"]) * iterations
    eval_s = existing_eval_per_job_s
    return {
        "generation_s": gen,
        "evaluation_s": eval_s,
        "optimization_s": opt,
        "total_s": gen + eval_s + opt,
        "total_min": (gen + eval_s + opt) / 60.0,
        "pes_50_jobs_h": (gen + eval_s + opt) * 50 / 3600.0,
    }


def smoke_job_rate(smoke_summary_path):
    s = load(smoke_summary_path)
    return {
        "iterations": s["config"]["training"]["iterations"],
        "total_s": s["timing_s"]["training_total"],
        "mean_iteration_s": s["timing_s"]["iteration_total"]["mean_s"],
        "projected_100_iter_s": s["timing_s"]["iteration_total"]["mean_s"] * 100,
        "projected_100_iter_min": s["timing_s"]["iteration_total"]["mean_s"] * 100 / 60,
        "projected_50_jobs_h": s["timing_s"]["iteration_total"]["mean_s"] * 100 * 50 / 3600,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mvp", default="runs/h4_mvp_gqkae_model_scale_benchmark.json")
    p.add_argument("--gqkae", default="runs/h4_paper_scale_gqkae_model_scale_benchmark.json")
    p.add_argument("--gqe", default="runs/h4_paper_scale_gqe_model_scale_benchmark.json")
    p.add_argument("--gqkae-smoke", default="runs/h4_paper_scale_gqkae_smoke_R0p88_seed7/summary.json")
    p.add_argument("--gqe-smoke", default="runs/h4_paper_scale_gqe_smoke_R0p88_seed7/summary.json")
    p.add_argument("--existing-summary", default="runs/h4_pes_cudaq_nvidia_summary.json")
    p.add_argument("--output", default="runs/h4_paper_faithful_eta_estimates.json")
    args = p.parse_args()

    existing = load(args.existing_summary)
    existing_rows = existing["rows"]
    eval_mean = statistics.fmean(r["train_timing_s"]["evaluation_total"]["total_s"] for r in existing_rows)
    job_mean = statistics.fmean(r["elapsed_s"] for r in existing_rows)

    payload = {
        "existing_mvp_cudaq": {
            "jobs": len(existing_rows),
            "mean_job_s": job_mean,
            "mean_job_min": job_mean / 60,
            "observed_50_job_pes_h": sum(r["elapsed_s"] for r in existing_rows) / 3600,
            "mean_evaluation_s_per_job": eval_mean,
        },
        "model_only_estimates": {
            "paper_scale_gqkae": model_job_estimate(load(args.gqkae), eval_mean),
            "paper_scale_gqe": model_job_estimate(load(args.gqe), eval_mean),
        },
        "smoke_projected_estimates": {
            "paper_scale_gqkae": smoke_job_rate(args.gqkae_smoke),
            "paper_scale_gqe": smoke_job_rate(args.gqe_smoke),
        },
        "notes": [
            "Model-only estimates add measured model generation+optimization to existing MVP CUDA-Q/QSCI evaluation cost.",
            "Smoke estimates use actual 3-iteration end-to-end CUDA-Q runner timing and extrapolate linearly to 100 iterations.",
            "Use smoke estimates as the primary local ETA because they include runner overhead and actual sampling/evaluation interactions.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
