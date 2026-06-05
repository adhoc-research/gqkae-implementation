"""Paper-style gate counting for generated GQKAE state-preparation circuits.

The paper reports counts assuming all-to-all connectivity and a standard Pauli-
evolution decomposition into arbitrary-angle single-qubit rotations, CX gates, and
single-qubit Clifford gates.  This matches the cited ``gqe-for-qsci`` implementation:
for each non-identity Pauli word of weight ``w``, emit basis-change Cliffords, a
CNOT ladder/uncompute, and one arbitrary ``rz``.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .chemistry import DeterminantBasis
from .fermion_mapping import PauliTerm, PauliWord, excitation_pauli_terms
from .operator_pool import OperatorPool

PAPER_GATE_KEYS = ("x", "h", "s", "sdg", "rz", "cx", "two_qubit", "total")


def paper_style_pauli_evolution_gate_count(word: PauliWord | str) -> Counter[str]:
    """Count the paper/cited-repo decomposition of one Pauli evolution.

    Per Pauli word, X basis changes use two H gates, Y basis changes use
    ``sdg``+``h`` before the CNOT ladder and ``h``+``s`` after it, and each
    non-identity word contributes one arbitrary-angle ``rz``.
    """
    letters = tuple(word)
    n_x = sum(p == "X" for p in letters)
    n_y = sum(p == "Y" for p in letters)
    n_z = sum(p == "Z" for p in letters)
    weight = n_x + n_y + n_z

    counts: Counter[str] = Counter()
    counts["cx"] = 0 if weight <= 1 else 2 * (weight - 1)
    counts["h"] = 2 * (n_x + n_y)
    counts["s"] = n_y
    counts["sdg"] = n_y
    counts["rz"] = 1 if weight >= 1 else 0
    counts["two_qubit"] = counts["cx"]
    counts["total"] = sum(counts[g] for g in ("cx", "h", "s", "sdg", "rz"))
    return counts


def paper_style_pauli_terms_gate_count(terms: Iterable[PauliTerm]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term in terms:
        if abs(term.coefficient) <= 1e-15:
            continue
        counts.update(paper_style_pauli_evolution_gate_count(term.word))
    counts["two_qubit"] = counts["cx"]
    counts["total"] = sum(counts[g] for g in ("x", "cx", "h", "s", "sdg", "rz"))
    return counts


def paper_style_sequence_gate_count(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
    include_hf_x: bool = True,
) -> dict[str, int]:
    """Return exact paper-style primitive counts for an operator sequence."""
    counts: Counter[str] = Counter()
    if include_hf_x:
        counts["x"] = sum(bit == "1" for bit in basis.hf_bitstring)
    for token in sequence:
        op = pool[int(token)]
        counts.update(paper_style_pauli_terms_gate_count(excitation_pauli_terms(op, basis.n_qubits)))
    counts["two_qubit"] = counts["cx"]
    counts["total"] = sum(counts[g] for g in ("x", "cx", "h", "s", "sdg", "rz"))
    return {key: int(counts.get(key, 0)) for key in PAPER_GATE_KEYS}


def normalize_paper_style_counts(counts: dict[str, int | float]) -> dict[str, int]:
    """Normalize a count mapping to the public paper-style keys."""
    out = {key: int(counts.get(key, 0)) for key in PAPER_GATE_KEYS}
    out["two_qubit"] = int(out.get("two_qubit") or out.get("cx", 0))
    out["total"] = int(sum(out[g] for g in ("x", "cx", "h", "s", "sdg", "rz")))
    return out
