#!/usr/bin/env python
"""Feasibility benchmark for an H4 VQE-style baseline.

This local feasibility script uses the same cited H4 Pauli-evolution pool as a
compact UCC-style variational ansatz and evaluates energies by exact statevector
projection into the H4 CAS determinant basis. It is intended to estimate optimizer
runtime and resource scale before deciding whether to run a more exact CUDA-Q/UCCSD
COBYLA baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from gqkae.chemistry import MoleculeConfig, build_h4_problem
from gqkae.circuits import _apply_pauli_evolution_full_state, _occ_int_to_full_index
from gqkae.config import OperatorPoolConfig
from gqkae.gate_counting import paper_style_sequence_gate_count
from gqkae.operator_pool import OperatorPool, build_operator_pool


def ansatz_state(theta: np.ndarray, pool: OperatorPool, n_qubits: int, hf_bitstring: str) -> np.ndarray:
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[_occ_int_to_full_index(hf_bitstring)] = 1.0
    # Skip token 0/noop; apply each pool operator once with an independent angle.
    for value, op in zip(theta, pool.operators[1:], strict=True):
        for coeff, word in op.pauli_terms:
            state = _apply_pauli_evolution_full_state(state, word, float(value) * float(coeff))
    norm = np.linalg.norm(state)
    return state / norm


def projected_energy(theta: np.ndarray, pool: OperatorPool, problem) -> float:
    full = ansatz_state(theta, pool, problem.n_qubits, problem.basis.hf_bitstring)
    active = np.zeros(problem.n_determinants, dtype=np.complex128)
    for idx, bitstring in enumerate(problem.basis.bitstrings):
        active[idx] = full[_occ_int_to_full_index(bitstring)]
    norm = np.linalg.norm(active)
    if norm < 1e-14:
        return 1e6
    active = active / norm
    return float(np.real(np.vdot(active, problem.hamiltonian @ active)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bond-length", type=float, default=0.88)
    p.add_argument("--maxiters", nargs="*", type=int, default=[10, 50, 100])
    p.add_argument("--project-to", type=int, default=5000)
    p.add_argument("--output", default="runs/h4_vqe_feasibility.json")
    args = p.parse_args()

    setup_start = time.perf_counter()
    mol_cfg = MoleculeConfig(bond_length_angstrom=args.bond_length, spin_orbital_ordering="interleaved")
    problem = build_h4_problem(mol_cfg)
    pool_cfg = OperatorPoolConfig(
        spec="paper_pauli_evolution",
        sequence_length=20,
        include_noop=True,
        remove_z_ladder=True,
        only_use_first_pauli=True,
        dedupe_pauli_words=True,
    )
    pool = build_operator_pool(problem, pool_cfg)
    setup_s = time.perf_counter() - setup_start

    n_params = pool.vocab_size - 1
    x0 = np.zeros(n_params, dtype=float)
    rows = []
    for maxiter in args.maxiters:
        calls = 0
        objective_times = []

        def objective(x):
            nonlocal calls
            calls += 1
            start = time.perf_counter()
            val = projected_energy(x, pool, problem)
            objective_times.append(time.perf_counter() - start)
            return val

        start = time.perf_counter()
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": int(maxiter), "rhobeg": 0.1})
        elapsed = time.perf_counter() - start
        x0 = np.asarray(res.x, dtype=float)
        mean_obj = float(np.mean(objective_times)) if objective_times else 0.0
        rows.append({
            "maxiter": int(maxiter),
            "objective_calls": int(calls),
            "elapsed_s": float(elapsed),
            "mean_objective_s": mean_obj,
            "best_energy": float(res.fun),
            "abs_error_vs_casci": abs(float(res.fun) - float(problem.casci_energy)),
            "projected_elapsed_s_for_requested_iters_linear": float(elapsed * args.project_to / max(maxiter, 1)),
            "projected_elapsed_s_from_objective_calls": float(mean_obj * args.project_to),
        })

    gate_count = paper_style_sequence_gate_count(list(range(1, pool.vocab_size)), pool, problem.basis)
    payload = {
        "method": "local UCC-style VQE feasibility using cited H4 Pauli-evolution pool and exact statevector objective",
        "note": "This estimates local optimizer/objective cost. A stricter CUDA-Q UCCSD/COBYLA baseline may differ and should be decided after ETA review.",
        "bond_length_angstrom": float(args.bond_length),
        "n_params": int(n_params),
        "setup_s": float(setup_s),
        "references": {
            "hf_energy": float(problem.hf_energy),
            "ccsd_energy": float(problem.ccsd_energy),
            "casci_energy": float(problem.casci_energy),
        },
        "gate_count_one_layer_all_nonnoop": gate_count,
        "project_to_iterations": int(args.project_to),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
