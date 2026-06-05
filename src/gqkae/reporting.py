"""Minimal MVP result reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def evaluate_run(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    best_path = run_path / "best.json"
    history_path = run_path / "history.json"
    if not summary_path.exists():
        if not best_path.exists():
            raise FileNotFoundError(f"no summary.json or best.json in {run_path}")
        return {"best": json.loads(best_path.read_text())}
    summary = json.loads(summary_path.read_text())
    if history_path.exists():
        summary["history_points"] = len(json.loads(history_path.read_text()))
    return summary


def format_report(summary: dict) -> str:
    refs = summary.get("references", {})
    model = summary.get("model", {})
    best = summary.get("best", {})
    final = summary.get("final", {})
    pool = summary.get("operator_pool", {})
    lines = ["GQKAE H4 MVP report"]
    if refs:
        lines.append(f"CASCI reference: {refs.get('casci_energy', float('nan')):.12f} Ha")
        lines.append(f"HF reference:    {refs.get('hf_energy', float('nan')):.12f} Ha")
        lines.append(f"Qubits/dets:     {refs.get('n_qubits')} / {refs.get('n_determinants')}")
    if pool:
        lines.append(f"Vocabulary/L:    {pool.get('vocab_size')} / {pool.get('sequence_length')}")
    if best:
        lines.append(f"Best energy:     {best.get('energy', float('nan')):.12f} Ha")
        lines.append(f"Best CASCI err:  {best.get('error', float('nan')):.6e} Ha")
        gate = best.get("gate_count", {}) or {}
        two_qubit = gate.get("two_qubit", gate.get("cx", "n/a"))
        label = "CUDA-Q gates" if "cx" in gate or "rz" in gate else "Approx gates"
        lines.append(f"{label}:    two-qubit={two_qubit}, total={gate.get('total', 'n/a')}")
    if final:
        err = final.get("final_batch_best_error")
        if err is not None:
            lines.append(f"Final CASCI err: {err:.6e} Ha")
    if model:
        lines.append(f"Parameters:      {model.get('parameters')} trainable")
        lines.append(f"Param memory:    {model.get('parameter_memory_mb', float('nan')):.3f} MB")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate/report an H4 GQKAE MVP run")
    parser.add_argument("--run-dir", default="runs/h4_mvp")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_run(args.run_dir)
    print(format_report(summary))


if __name__ == "__main__":  # pragma: no cover
    main()
