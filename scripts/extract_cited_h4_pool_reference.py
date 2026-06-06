#!/usr/bin/env python
"""Extract the cited Tequila/CUDA-Q-style H4 PauliEvolutionPool reference.

This script mirrors the operator-pool construction in
`moken20/gqe-for-qsci/gqe_qsci/gqe/operator_pool.py` without requiring that
repository to be installed. It intentionally uses Tequila for the canonical
excitation-gate -> Pauli-string path, because the final H4 replication claim
should be based on the cited method rather than this repo's local fallback
mapping.

Output JSON contains the vocabulary order, Pauli words, coefficients/gate
parameters, and paper-style per-token gate counts. It can be regenerated with:

    uv run --extra paper python scripts/extract_cited_h4_pool_reference.py \
      --output data/h4_cited_pauli_evolution_pool.json

The default settings mirror the cited default config: params=null,
ccsd_threshold=1e-6, remove_z_ladder=true, only_use_first_pauli=true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require_tequila():
    try:
        import tequila as tq
        from tequila.circuit import QCircuit
        from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Tequila is required for this reference extractor. "
            "Install the optional paper dependencies, e.g. `uv sync --extra paper`."
        ) from exc
    return tq, QCircuit, ClosedShellAmplitudes


def paper_style_pauli_evolution_gate_count(pauli: str) -> dict[str, int]:
    counter = Counter(pauli)
    n_x = counter.get("X", 0)
    n_y = counter.get("Y", 0)
    n_z = counter.get("Z", 0)
    weight = n_x + n_y + n_z
    out = Counter()
    out["cx"] = 0 if weight <= 1 else 2 * (weight - 1)
    out["h"] = 2 * (n_x + n_y)
    out["s"] = n_y
    out["sdg"] = n_y
    out["rz"] = 1 if weight >= 1 else 0
    out["two_qubit"] = out["cx"]
    out["total"] = sum(out[g] for g in ("cx", "h", "s", "sdg", "rz"))
    return {k: int(out.get(k, 0)) for k in ("h", "s", "sdg", "rz", "cx", "two_qubit", "total")}


@dataclass(frozen=True)
class H4CitedMolecule:
    geometry: list[list[Any]]
    basis: str
    nelec: tuple[int, int]
    norb: int
    active_indices: list[int]
    ccsd_amplitude: dict[str, np.ndarray]


def make_linear_h4_geometry(bond_length: float) -> list[list[Any]]:
    # Cited helper places a linear chain along z. Orientation should not affect RHF/CCSD
    # energies, but we mirror the cited repository exactly here.
    return [["H", [0.0, 0.0, i * float(bond_length)]] for i in range(4)]


def build_cited_h4_molecule(bond_length: float, basis: str, active_electrons: int, active_orbitals: int) -> H4CitedMolecule:
    try:
        from pyscf import cc, gto, lib, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PySCF is required to extract the H4 cited pool reference.") from exc

    lib.num_threads(1)
    geometry = make_linear_h4_geometry(bond_length)
    mol = gto.M(atom=geometry, basis=basis, charge=0, spin=0, unit="Angstrom", verbose=0)
    hf = scf.RHF(mol).run(verbose=0)
    nelec = (active_electrons // 2, active_electrons // 2)
    mc = mcscf.CASCI(hf, active_orbitals, nelec)
    ncore = (mol.nelectron - sum(nelec)) // 2
    active_indices = list(range(ncore, ncore + active_orbitals))
    nmo = hf.mo_coeff.shape[1]
    active_set = set(active_indices)
    frozen_orbs = [i for i in range(nmo) if i not in active_set]
    mycc = cc.RCCSD(hf, frozen=frozen_orbs)
    mycc.verbose = 0
    mycc.kernel()
    return H4CitedMolecule(
        geometry=geometry,
        basis=basis,
        nelec=nelec,
        norb=active_orbitals,
        active_indices=active_indices,
        ccsd_amplitude={"t1": np.asarray(mycc.t1), "t2": np.asarray(mycc.t2)},
    )


def generate_excitations(cited_mol: H4CitedMolecule, threshold: float) -> dict[tuple[int, ...], float]:
    """Mirror UCCSDBasedPool.generate_excitations from the cited repository."""
    _tq, _QCircuit, ClosedShellAmplitudes = _require_tequila()
    ccsd_amplitudes = ClosedShellAmplitudes(
        tIjAb=cited_mol.ccsd_amplitude["t2"],
        tIA=cited_mol.ccsd_amplitude["t1"],
    )
    amplitudes_all = ccsd_amplitudes.make_parameter_dictionary(threshold=0.0, screening=False)
    amplitudes = {
        k: v for k, v in amplitudes_all.items()
        if not np.isclose(v, 0.0, atol=threshold)
    }
    amplitudes = dict(sorted(amplitudes.items(), key=lambda x: np.fabs(x[1]), reverse=True))
    indices: dict[tuple[int, ...], float] = {}
    for key, t in amplitudes.items():
        assert len(key) % 2 == 0
        if not np.isclose(t, 0.0, atol=threshold):
            if len(key) == 2:
                angle = 2.0 * t
                idx_a = (2 * key[0], 2 * key[1])
                idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                indices[idx_a] = angle
                indices[idx_b] = angle
            else:
                assert len(key) == 4
                angle = 2.0 * t
                idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                indices[idx_abab] = angle
                if key[0] != key[2] and key[1] != key[3]:
                    idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                    idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                    partner = tuple([key[2], key[1], key[0], key[3]])
                    partner_t = amplitudes_all.get(partner, 0.0)
                    anglex = 2.0 * (t - partner_t)
                    indices[idx_aaaa] = anglex
                    indices[idx_bbbb] = anglex
    return indices


def _paulistring_to_dict(paulistring: Any) -> dict[int, str]:
    # Tequila PauliString behaves like a mapping from qubit index -> Pauli letter.
    return {int(k): str(v).upper() for k, v in paulistring.items()}


def _pauli_dict_to_word(pauli: dict[int, str], n_qubits: int) -> str:
    letters = ["I"] * n_qubits
    for k, v in pauli.items():
        if v.upper() in {"X", "Y", "Z"}:
            letters[int(k)] = v.upper()
    return "".join(letters)


def build_tequila_ansatz(cited_mol: H4CitedMolecule, excitations: dict[tuple[int, ...], float]):
    tq, QCircuit, _ClosedShellAmplitudes = _require_tequila()
    geometry_lines = [f"{atom_type} {coords[0]} {coords[1]} {coords[2]}" for atom_type, coords in cited_mol.geometry]
    geometry_str = "\n".join(geometry_lines)
    tq_molecule = tq.Molecule(
        geometry=geometry_str,
        basis_set=cited_mol.basis,
        active_orbitals=cited_mol.active_indices,
        transformation="jordan-wigner",
    )
    ansatz = QCircuit()
    for idx, angle in excitations.items():
        converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
        ansatz += tq_molecule.make_excitation_gate(indices=converted, angle=angle)
    return ansatz


def extract_pool(
    bond_length: float,
    basis: str,
    active_electrons: int,
    active_orbitals: int,
    threshold: float,
    remove_z_ladder: bool,
    only_use_first_pauli: bool,
    include_identity: bool,
) -> dict[str, Any]:
    cited_mol = build_cited_h4_molecule(bond_length, basis, active_electrons, active_orbitals)
    excitations = generate_excitations(cited_mol, threshold=threshold)
    ansatz = build_tequila_ansatz(cited_mol, excitations)
    n_qubits = active_orbitals * 2

    tokens: list[dict[str, Any]] = []
    if include_identity:
        tokens.append({
            "token": 0,
            "kind": "identity",
            "pauli_word": "I" * n_qubits,
            "coefficient": 0.0,
            "tequila_gate_parameter": 0.0,
            "tequila_paulistring_coeff": 0.0,
            "source_gate_index": None,
            "gate_count": paper_style_pauli_evolution_gate_count("I" * n_qubits),
        })

    seen: set[str] = set()
    for gate_index, gate in enumerate(ansatz.gates):
        coeff = float(gate.parameter)
        for p_index, p in enumerate(gate.generator.paulistrings):
            pauli = _paulistring_to_dict(p)
            if remove_z_ladder:
                pauli = {k: v for k, v in pauli.items() if v.lower() != "z"}
            word = _pauli_dict_to_word(pauli, n_qubits)
            if word in seen:
                continue
            seen.add(word)
            tokens.append({
                "token": len(tokens),
                "kind": "pauli_evolution",
                "pauli_word": word,
                # Cited PauliEvolutionPool with params=null appends coeff * SpinOperator(term).
                "coefficient": coeff,
                "tequila_gate_parameter": coeff,
                "tequila_paulistring_coeff": float(getattr(p, "_coeff", 1.0)),
                "source_gate_index": gate_index,
                "source_paulistring_index": p_index,
                "gate_count": paper_style_pauli_evolution_gate_count(word),
            })
            if only_use_first_pauli:
                break

    payload: dict[str, Any] = {
        "schema": "gqkae.h4_cited_pauli_evolution_pool.v1",
        "settings": {
            "bond_length_angstrom": bond_length,
            "basis": basis,
            "active_electrons": active_electrons,
            "active_orbitals": active_orbitals,
            "n_qubits": n_qubits,
            "ccsd_threshold": threshold,
            "remove_z_ladder": remove_z_ladder,
            "only_use_first_pauli": only_use_first_pauli,
            "params": None,
            "include_identity": include_identity,
            "spin_orbital_ordering": "interleaved_alpha_beta",
        },
        "active_indices": cited_mol.active_indices,
        "n_screened_spin_excitations": len(excitations),
        "screened_spin_excitations": [
            {"indices": list(k), "coefficient": float(v)} for k, v in excitations.items()
        ],
        "vocab_size": len(tokens),
        "tokens": tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=0.88)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--active-electrons", type=int, default=4)
    parser.add_argument("--active-orbitals", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--remove-z-ladder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-use-first-pauli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("data/h4_cited_pauli_evolution_pool.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = extract_pool(
        bond_length=args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        threshold=args.threshold,
        remove_z_ladder=args.remove_z_ladder,
        only_use_first_pauli=args.only_use_first_pauli,
        include_identity=args.include_identity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "vocab_size": payload["vocab_size"],
        "n_screened_spin_excitations": payload["n_screened_spin_excitations"],
        "sha256": payload["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
