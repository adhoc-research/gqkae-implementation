"""Sequence-to-circuit and simulation utilities.

CUDA-Q is the paper-fidelity target backend. The MVP also contains a determinant-space
state-vector simulator over the H4 CASCI basis so tests and local smoke runs can execute
on machines without CUDA-Q. The public API is backend-oriented, making replacement with a
full CUDA-Q fermionic-excitation decomposition straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .chemistry import DeterminantBasis
from .fermion_mapping import excitation_pauli_terms
from .gate_counting import paper_style_sequence_gate_count
from .operator_pool import ExcitationOperator, OperatorPool


@dataclass(frozen=True)
class CircuitSampleResult:
    counts: dict[str, int]
    probabilities: dict[str, float]
    gate_count: dict[str, int]
    backend: str


@dataclass(frozen=True)
class CudaQCircuitSource:
    """Serializable CUDA-Q kernel source for inspecting/generated-circuit handoff."""

    source: str
    gate_count: dict[str, int]


def hf_state_vector(basis: DeterminantBasis) -> np.ndarray:
    state = np.zeros(basis.size, dtype=np.complex128)
    state[basis.index_by_bitstring[basis.hf_bitstring]] = 1.0
    return state


def _apply_annihilate(occ: int, orbital: int) -> tuple[int, int] | None:
    if ((occ >> orbital) & 1) == 0:
        return None
    parity = (occ & ((1 << orbital) - 1)).bit_count()
    return occ ^ (1 << orbital), -1 if parity % 2 else 1


def _apply_create(occ: int, orbital: int) -> tuple[int, int] | None:
    if ((occ >> orbital) & 1) == 1:
        return None
    parity = (occ & ((1 << orbital) - 1)).bit_count()
    return occ | (1 << orbital), -1 if parity % 2 else 1


def apply_excitation_to_occ(
    occ: int,
    annihilate: tuple[int, ...],
    create: tuple[int, ...],
) -> tuple[int, int] | None:
    """Apply product of creators/annihilators to a spin-orbital occupation integer.

    Operators are applied right-to-left: annihilate occupied orbitals, then create
    virtual orbitals. Returns the new occupation and fermionic sign.
    """
    sign = 1
    cur = occ
    for orbital in annihilate:
        out = _apply_annihilate(cur, orbital)
        if out is None:
            return None
        cur, s = out
        sign *= s
    for orbital in reversed(create):
        out = _apply_create(cur, orbital)
        if out is None:
            return None
        cur, s = out
        sign *= s
    return cur, sign


def _bitstring_to_occ_int(bitstring: str) -> int:
    return sum((ch == "1") << i for i, ch in enumerate(bitstring))


def _occ_int_to_bitstring(occ: int, n_qubits: int) -> str:
    return "".join("1" if ((occ >> i) & 1) else "0" for i in range(n_qubits))


def apply_excitation_rotation(
    state: np.ndarray,
    op: ExcitationOperator,
    basis: DeterminantBasis,
) -> np.ndarray:
    """Apply exp(theta * (T - T^dagger)) in the determinant basis."""
    if op.is_noop:
        return state.copy()

    out = state.copy()
    visited: set[tuple[int, int]] = set()
    c = np.cos(op.angle)
    s = np.sin(op.angle)

    for idx, bitstring in enumerate(basis.bitstrings):
        occ = _bitstring_to_occ_int(bitstring)
        partner = apply_excitation_to_occ(occ, op.annihilate, op.create)
        if partner is None:
            continue
        partner_occ, sign = partner
        partner_bitstring = _occ_int_to_bitstring(partner_occ, basis.n_qubits)
        jdx = basis.index_by_bitstring.get(partner_bitstring)
        if jdx is None:
            continue
        key = tuple(sorted((idx, jdx)))
        if key in visited:
            continue
        visited.add(key)

        a = state[idx]
        b = state[jdx]
        out[idx] = c * a - sign * s * b
        out[jdx] = sign * s * a + c * b
    return out


def _occ_int_to_full_index(bitstring: str) -> int:
    return sum((ch == "1") << i for i, ch in enumerate(bitstring))


def _full_index_to_bitstring(index: int, n_qubits: int) -> str:
    return "".join("1" if ((index >> i) & 1) else "0" for i in range(n_qubits))


def _apply_pauli_word_to_full_state(state: np.ndarray, word: tuple[str, ...]) -> np.ndarray:
    n_qubits = len(word)
    out = np.zeros_like(state)
    for idx, amp in enumerate(state):
        if abs(amp) <= 1e-15:
            continue
        target = idx
        phase = 1.0 + 0.0j
        for q, pauli in enumerate(word):
            bit = (idx >> q) & 1
            if pauli == "I":
                continue
            if pauli == "X":
                target ^= 1 << q
            elif pauli == "Y":
                target ^= 1 << q
                phase *= 1j if bit == 0 else -1j
            elif pauli == "Z":
                phase *= 1 if bit == 0 else -1
            else:
                raise ValueError(f"unknown Pauli letter {pauli!r}")
        out[target] += phase * amp
    return out


def _apply_pauli_evolution_full_state(state: np.ndarray, word: tuple[str, ...], angle: float) -> np.ndarray:
    return np.cos(angle) * state - 1j * np.sin(angle) * _apply_pauli_word_to_full_state(state, word)


def simulate_full_qubit_state_vector(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
) -> np.ndarray:
    state = np.zeros(1 << basis.n_qubits, dtype=np.complex128)
    state[_occ_int_to_full_index(basis.hf_bitstring)] = 1.0
    for token in sequence:
        op = pool[int(token)]
        for term in excitation_pauli_terms(op, basis.n_qubits):
            state = _apply_pauli_evolution_full_state(state, term.word, op.angle * term.coefficient)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise FloatingPointError("zero full-qubit state generated by Pauli-evolution sequence")
    return state / norm


def simulate_state_vector(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
) -> np.ndarray:
    if any(getattr(pool[int(token)], "pauli_terms", ()) for token in sequence):
        full_state = simulate_full_qubit_state_vector(sequence, pool, basis)
        state = np.zeros(basis.size, dtype=np.complex128)
        for idx, bitstring in enumerate(basis.bitstrings):
            state[idx] = full_state[_occ_int_to_full_index(bitstring)]
        norm = np.linalg.norm(state)
        if norm == 0:
            raise FloatingPointError("Pauli-evolution sequence has zero amplitude in determinant basis")
        return state / norm

    state = hf_state_vector(basis)
    for token in sequence:
        state = apply_excitation_rotation(state, pool[int(token)], basis)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise FloatingPointError("zero state generated by excitation sequence")
    return state / norm


def sample_counts_from_state(
    state: np.ndarray,
    basis: DeterminantBasis,
    shots: int,
    rng: np.random.Generator,
) -> tuple[dict[str, int], dict[str, float]]:
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    sampled = rng.multinomial(int(shots), probs)
    counts = {
        basis.bitstrings[i]: int(c)
        for i, c in enumerate(sampled)
        if int(c) > 0
    }
    probabilities = {basis.bitstrings[i]: float(p) for i, p in enumerate(probs) if p > 1e-15}
    return counts, probabilities


def sequence_to_cudaq_source(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
) -> CudaQCircuitSource:
    """Convert an operator sequence into exact paper-style CUDA-Q source.

    The generated source initializes the HF determinant and expands each UCCSD token into
    Jordan-Wigner Pauli evolutions using the paper/cited-repo all-to-all decomposition:
    Clifford basis changes, a CX ladder/uncompute, and one arbitrary ``rz`` per Pauli
    word.
    """
    lines = [
        "import cudaq",
        "",
        "@cudaq.kernel",
        "def h4_sequence_kernel():",
        f"    q = cudaq.qvector({basis.n_qubits})",
    ]
    hf_occ = [i for i, bit in enumerate(basis.hf_bitstring) if bit == "1"]
    for qubit in hf_occ:
        lines.append(f"    x(q[{qubit}])")

    for token in sequence:
        op = pool[int(token)]
        terms = excitation_pauli_terms(op, basis.n_qubits)
        if not terms:
            continue
        lines.append(f"    # token {int(token)}: {op.name}")
        for term in terms:
            active = [i for i, pauli in enumerate(term.word) if pauli != "I"]
            if not active or abs(term.coefficient) <= 1e-15:
                continue
            word = "".join(term.word)
            angle = 2.0 * op.angle * term.coefficient
            lines.append(f"    # exp Pauli {word}, coeff={term.coefficient:.17g}")
            for q in active:
                pauli = term.word[q]
                if pauli == "X":
                    lines.append(f"    h(q[{q}])")
                elif pauli == "Y":
                    lines.append(f"    sdg(q[{q}])")
                    lines.append(f"    h(q[{q}])")
            target = active[-1]
            for q in active[:-1]:
                lines.append(f"    cx(q[{q}], q[{target}])")
            lines.append(f"    rz({angle:.17g}, q[{target}])")
            for q in reversed(active[:-1]):
                lines.append(f"    cx(q[{q}], q[{target}])")
            for q in reversed(active):
                pauli = term.word[q]
                if pauli == "X":
                    lines.append(f"    h(q[{q}])")
                elif pauli == "Y":
                    lines.append(f"    h(q[{q}])")
                    lines.append(f"    s(q[{q}])")
    lines.append("    mz(q)")
    return CudaQCircuitSource(
        "\n".join(lines) + "\n",
        paper_style_sequence_gate_count(sequence, pool, basis),
    )


def sample_sequence_circuit(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
    shots: int,
    rng: np.random.Generator,
    backend: str = "determinant",
    cudaq_target: str | None = None,
    cudaq_option: str | None = None,
    cudaq_seed: int | None = None,
    cudaq_reverse_bitstrings: bool = False,
) -> CircuitSampleResult:
    """Sample measurement counts for a generated operator sequence."""
    if backend == "cudaq":
        return _sample_with_cudaq_or_raise(
            sequence,
            pool,
            basis,
            shots,
            rng,
            target=cudaq_target,
            option=cudaq_option,
            seed=cudaq_seed,
            reverse_bitstrings=cudaq_reverse_bitstrings,
        )
    if backend != "determinant":
        raise ValueError(f"unknown backend {backend!r}; expected determinant or cudaq")
    state = simulate_state_vector(sequence, pool, basis)
    counts, probabilities = sample_counts_from_state(state, basis, shots, rng)
    return CircuitSampleResult(
        counts=counts,
        probabilities=probabilities,
        gate_count=paper_style_sequence_gate_count(sequence, pool, basis),
        backend="determinant",
    )


def _sample_with_cudaq_or_raise(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
    shots: int,
    rng: np.random.Generator,
    target: str | None = None,
    option: str | None = None,
    seed: int | None = None,
    reverse_bitstrings: bool = False,
) -> CircuitSampleResult:
    """Sample a generated sequence with a real CUDA-Q kernel backend."""
    from .cudaq_backend import sample_cudaq_sequence

    del rng  # CUDA-Q owns backend sampling randomness once its seed is configured.
    result = sample_cudaq_sequence(
        sequence,
        pool,
        basis,
        shots=shots,
        target=target,
        option=option,
        seed=seed,
        reverse_bitstrings=reverse_bitstrings,
    )
    return CircuitSampleResult(
        counts=result.counts,
        probabilities=result.probabilities,
        gate_count=result.resources,
        backend="cudaq",
    )
