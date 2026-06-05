"""Fermion-to-Pauli mapping utilities for CUDA-Q circuit emission.

This module implements the Jordan-Wigner transform directly to avoid adding another
heavy dependency. The convention matches the repository's internal qubit ordering:
spin-orbital index ``j`` maps to qubit ``j`` and

    a_j^† = Z_0 ... Z_{j-1} (X_j - i Y_j) / 2
    a_j   = Z_0 ... Z_{j-1} (X_j + i Y_j) / 2

For an excitation operator T, G = T - T^† is anti-Hermitian and H = iG is Hermitian.
The CUDA-Q backend emits exp(-i theta H), matching exp(theta G).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .operator_pool import ExcitationOperator

PauliWord = tuple[str, ...]


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    word: PauliWord


_PAULI_MUL: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("Y", "I"): (1, "Y"),
    ("Z", "I"): (1, "Z"),
    ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"),
    ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def _mul_words(a: PauliWord, b: PauliWord) -> tuple[complex, PauliWord]:
    phase = 1 + 0j
    out: list[str] = []
    for pa, pb in zip(a, b, strict=True):
        ph, pc = _PAULI_MUL[(pa, pb)]
        phase *= ph
        out.append(pc)
    return phase, tuple(out)


def _jw_ladder(index: int, creation: bool, n_qubits: int) -> list[tuple[complex, PauliWord]]:
    z_prefix = ["Z" if q < index else "I" for q in range(n_qubits)]
    x_word = list(z_prefix)
    y_word = list(z_prefix)
    x_word[index] = "X"
    y_word[index] = "Y"
    y_coeff = -0.5j if creation else 0.5j
    return [(0.5 + 0j, tuple(x_word)), (y_coeff, tuple(y_word))]


def _multiply_operator_terms(
    left: list[tuple[complex, PauliWord]],
    right: list[tuple[complex, PauliWord]],
) -> list[tuple[complex, PauliWord]]:
    out: dict[PauliWord, complex] = {}
    for ca, wa in left:
        for cb, wb in right:
            ph, word = _mul_words(wa, wb)
            out[word] = out.get(word, 0j) + ca * cb * ph
    return [(c, w) for w, c in out.items() if abs(c) > 1e-14]


def jordan_wigner_monomial(
    ladder_ops: Iterable[tuple[int, bool]],
    n_qubits: int,
) -> dict[PauliWord, complex]:
    """Map a product of fermionic ladder operators to Pauli terms.

    ``ladder_ops`` is in left-to-right operator order, e.g.
    ``a_a^† a_i`` is ``[(a, True), (i, False)]``.
    """
    terms = [(1 + 0j, tuple("I" for _ in range(n_qubits)))]
    for index, creation in ladder_ops:
        terms = _multiply_operator_terms(terms, _jw_ladder(index, creation, n_qubits))
    return {word: coeff for coeff, word in terms if abs(coeff) > 1e-14}


def _adjoint_ladder_ops(ladder_ops: tuple[tuple[int, bool], ...]) -> tuple[tuple[int, bool], ...]:
    return tuple((idx, not creation) for idx, creation in reversed(ladder_ops))


def excitation_ladder_ops(op: ExcitationOperator) -> tuple[tuple[int, bool], ...]:
    """Return T for an excitation token in the simulator's fermionic sign convention."""
    return tuple((idx, True) for idx in op.create) + tuple(
        (idx, False) for idx in reversed(op.annihilate)
    )


def excitation_pauli_terms(op: ExcitationOperator, n_qubits: int) -> tuple[PauliTerm, ...]:
    """Return Hermitian H = i(T - T^†) as real Pauli terms."""
    if op.is_noop:
        return ()
    ladder = excitation_ladder_ops(op)
    mapped_t = jordan_wigner_monomial(ladder, n_qubits)
    mapped_t_dag = jordan_wigner_monomial(_adjoint_ladder_ops(ladder), n_qubits)
    combined: dict[PauliWord, complex] = {}
    for word, coeff in mapped_t.items():
        combined[word] = combined.get(word, 0j) + 1j * coeff
    for word, coeff in mapped_t_dag.items():
        combined[word] = combined.get(word, 0j) - 1j * coeff

    terms: list[PauliTerm] = []
    for word, coeff in combined.items():
        if abs(coeff) <= 1e-12:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"non-Hermitian mapped coefficient for {op.name}: {coeff}")
        terms.append(PauliTerm(float(coeff.real), word))
    return tuple(sorted(terms, key=lambda t: "".join(t.word)))


def pauli_terms_matrix(terms: Iterable[PauliTerm]) -> np.ndarray:
    mats = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    out: np.ndarray | None = None
    for term in terms:
        mat = np.array([[1]], dtype=complex)
        # Internal state-vector index uses qubit 0 as least significant, so reverse
        # tensor order for conventional dense matrices indexed by integer bit value.
        for p in reversed(term.word):
            mat = np.kron(mat, mats[p])
        out = term.coefficient * mat if out is None else out + term.coefficient * mat
    if out is None:
        return np.zeros((1, 1), dtype=complex)
    return out
