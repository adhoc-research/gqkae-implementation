"""Diagnostics for matching H4 GQKAE circuit counts to the paper/cited setup."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Iterable

from gqkae.chemistry import make_determinant_basis
from gqkae.config import OperatorPoolConfig
from gqkae.fermion_mapping import PauliTerm, excitation_pauli_terms
from gqkae.gate_counting import paper_style_pauli_evolution_gate_count, paper_style_sequence_gate_count
from gqkae.operator_pool import build_h4_uccsd_pool


def _strip_z(word: Iterable[str]) -> tuple[str, ...]:
    return tuple("I" if p == "Z" else p for p in word)


def _variant_terms(terms: tuple[PauliTerm, ...], remove_z_ladder: bool, only_first_pauli: bool) -> tuple[PauliTerm, ...]:
    selected = terms[:1] if only_first_pauli else terms
    if not remove_z_ladder:
        return selected
    return tuple(PauliTerm(t.coefficient, _strip_z(t.word)) for t in selected)


def _count_terms(terms: Iterable[PauliTerm], include_hf_x: int = 0) -> dict[str, int]:
    counts: Counter[str] = Counter()
    counts["x"] = include_hf_x
    for term in terms:
        counts.update(paper_style_pauli_evolution_gate_count(term.word))
    counts["two_qubit"] = counts["cx"]
    counts["total"] = sum(counts[g] for g in ("x", "cx", "h", "s", "sdg", "rz"))
    return {k: int(counts.get(k, 0)) for k in ("x", "h", "s", "sdg", "rz", "cx", "two_qubit", "total")}


def _count_sequence(sequence: list[int], pool, basis, remove_z_ladder: bool, only_first_pauli: bool) -> dict[str, int]:
    if not remove_z_ladder and not only_first_pauli:
        return paper_style_sequence_gate_count(sequence, pool, basis)
    all_terms: list[PauliTerm] = []
    for token in sequence:
        all_terms.extend(
            _variant_terms(
                excitation_pauli_terms(pool[int(token)], basis.n_qubits),
                remove_z_ladder=remove_z_ladder,
                only_first_pauli=only_first_pauli,
            )
        )
    return _count_terms(all_terms, include_hf_x=sum(bit == "1" for bit in basis.hf_bitstring))


def _load_sequence(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    if "best" in data:
        return [int(x) for x in data["best"]["sequence"]]
    return [int(x) for x in data["sequence"]]


def build_diagnostics(best_paths: list[Path]) -> dict:
    basis = make_determinant_basis(4, 2, 2)
    pool = build_h4_uccsd_pool(basis, OperatorPoolConfig(sequence_length=20, include_noop=True))
    variants = {
        "full_excitation_terms": {"remove_z_ladder": False, "only_first_pauli": False},
        "full_terms_remove_z_ladder": {"remove_z_ladder": True, "only_first_pauli": False},
        "first_pauli_only": {"remove_z_ladder": False, "only_first_pauli": True},
        "first_pauli_remove_z_ladder": {"remove_z_ladder": True, "only_first_pauli": True},
    }
    sequences = {}
    for path in best_paths:
        if path.exists():
            seq = _load_sequence(path)
            sequences[str(path)] = {
                "sequence": seq,
                "variant_counts": {
                    name: _count_sequence(seq, pool, basis, **opts) for name, opts in variants.items()
                },
            }
    token_table = []
    for token, op in enumerate(pool.operators):
        terms = excitation_pauli_terms(op, basis.n_qubits)
        row = {"token": token, "name": op.name, "n_pauli_terms": len(terms)}
        for name, opts in variants.items():
            row[name] = _count_terms(_variant_terms(terms, **opts), include_hf_x=0)
        token_table.append(row)
    return {
        "paper_table_i_h4_gqkae": {"two_qubit_mean": 100.0, "two_qubit_std": 3.7, "total_mean": 314.0, "total_std": 15.0},
        "sequences": sequences,
        "token_table": token_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", action="append", default=["runs/h4_cudaq_paper_like/summary.json"])
    parser.add_argument("--output", default="runs/h4_gate_match_diagnostics/diagnostics.json")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = build_diagnostics([Path(p) for p in args.best])
    out.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics["sequences"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
