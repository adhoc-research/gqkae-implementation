"""CUDA-Q backend for H4 GQKAE sequence sampling.

The backend emits Jordan-Wigner Pauli rotations for each UCCSD excitation token and
uses CUDA-Q for sampling and resource estimation when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .chemistry import DeterminantBasis
from .fermion_mapping import PauliTerm, excitation_pauli_terms
from .gate_counting import paper_style_sequence_gate_count
from .operator_pool import OperatorPool


@dataclass(frozen=True)
class CudaQExecutionResult:
    counts: dict[str, int]
    probabilities: dict[str, float]
    resources: dict[str, int | float | dict]


def _import_cudaq():
    try:
        import cudaq
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "qsci.backend=cudaq requested, but CUDA-Q is not installed. "
            "Install with `uv sync --extra cuda` on a supported Linux system."
        ) from exc
    return cudaq


def configure_cudaq_target(target: str | None = None, option: str | None = None, seed: int | None = None) -> None:
    cudaq = _import_cudaq()
    if target:
        if option:
            cudaq.set_target(target, option=option)
        else:
            cudaq.set_target(target)
    if seed is not None and hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(int(seed))


def _x(kernel, qubit) -> None:
    kernel.x(qubit)


def _h(kernel, qubit) -> None:
    kernel.h(qubit)


def _rx(kernel, angle: float, qubit) -> None:
    kernel.rx(float(angle), qubit)


def _rz(kernel, angle: float, qubit) -> None:
    kernel.rz(float(angle), qubit)


def _s(kernel, qubit) -> None:
    gate = getattr(kernel, "s", None)
    if callable(gate):
        gate(qubit)
    else:  # pragma: no cover - depends on CUDA-Q builder version
        _rz(kernel, pi / 2, qubit)


def _sdg(kernel, qubit) -> None:
    gate = getattr(kernel, "sdg", None)
    if callable(gate):
        gate(qubit)
    else:  # pragma: no cover - depends on CUDA-Q builder version
        _rz(kernel, -pi / 2, qubit)


def _cx(kernel, control, target) -> None:
    kernel.cx(control, target)


def _mz(kernel, qubits) -> None:
    kernel.mz(qubits)


def _apply_basis_change(kernel, qubit, pauli: str) -> None:
    if pauli == "X":
        _h(kernel, qubit)
    elif pauli == "Y":
        # Paper/cited-repo Clifford convention for Y basis: S† then H.
        _sdg(kernel, qubit)
        _h(kernel, qubit)


def _undo_basis_change(kernel, qubit, pauli: str) -> None:
    if pauli == "X":
        _h(kernel, qubit)
    elif pauli == "Y":
        # Undo H S† as H then S.
        _h(kernel, qubit)
        _s(kernel, qubit)


def apply_pauli_exponential(kernel, qubits, term: PauliTerm, theta: float) -> None:
    """Emit exp(-i theta * coefficient * P) into a CUDA-Q kernel builder."""
    active = [i for i, p in enumerate(term.word) if p != "I"]
    if not active or abs(term.coefficient) <= 1e-15:
        return
    angle = float(theta * term.coefficient)
    for q in active:
        _apply_basis_change(kernel, qubits[q], term.word[q])
    target = active[-1]
    for q in active[:-1]:
        _cx(kernel, qubits[q], qubits[target])
    _rz(kernel, 2.0 * angle, qubits[target])
    for q in reversed(active[:-1]):
        _cx(kernel, qubits[q], qubits[target])
    for q in reversed(active):
        _undo_basis_change(kernel, qubits[q], term.word[q])


def build_cudaq_kernel(sequence: list[int] | np.ndarray, pool: OperatorPool, basis: DeterminantBasis):
    """Build a CUDA-Q kernel for a generated operator-token sequence."""
    cudaq = _import_cudaq()
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(basis.n_qubits)

    for q, bit in enumerate(basis.hf_bitstring):
        if bit == "1":
            _x(kernel, qubits[q])

    for token in sequence:
        op = pool[int(token)]
        for term in excitation_pauli_terms(op, basis.n_qubits):
            apply_pauli_exponential(kernel, qubits, term, op.angle)

    _mz(kernel, qubits)
    return kernel


def _normalize_sample_key(bitstring: str, n_qubits: int, reverse_bitstrings: bool = False) -> str:
    clean = str(bitstring).replace(" ", "")
    if len(clean) != n_qubits:
        # CUDA-Q registers can include decorations for some targets; keep only bits.
        clean = "".join(ch for ch in clean if ch in "01")
    if len(clean) != n_qubits:
        raise ValueError(f"CUDA-Q returned bitstring {bitstring!r}, expected {n_qubits} bits")
    return clean[::-1] if reverse_bitstrings else clean


def _sample_result_to_counts(sample_result, n_qubits: int, reverse_bitstrings: bool) -> dict[str, int]:
    return {
        _normalize_sample_key(bitstring, n_qubits, reverse_bitstrings): int(count)
        for bitstring, count in sample_result.items()
        if int(count) > 0
    }


def _resources_to_dict(resources) -> dict[str, int | float | dict]:
    if hasattr(resources, "to_dict"):
        data = dict(resources.to_dict())
    else:
        data = {}
    for attr in (
        "depth",
        "multi_qubit_depth",
        "multi_qubit_gate_count",
        "num_qubits",
        "num_used_qubits",
        "gate_count_by_arity",
    ):
        if hasattr(resources, attr):
            value = getattr(resources, attr)
            data[attr] = value() if callable(value) else value
    if hasattr(resources, "count"):
        try:
            data["total"] = int(resources.count())
        except TypeError:
            pass
    if "multi_qubit_gate_count" in data:
        data["two_qubit"] = int(data["multi_qubit_gate_count"])
    elif "cx" in data:
        data["two_qubit"] = int(data["cx"])
    return data


def sample_cudaq_sequence(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
    shots: int,
    target: str | None = None,
    option: str | None = None,
    seed: int | None = None,
    reverse_bitstrings: bool = False,
) -> CudaQExecutionResult:
    cudaq = _import_cudaq()
    configure_cudaq_target(target=target, option=option, seed=seed)
    kernel = build_cudaq_kernel(sequence, pool, basis)
    sample_result = cudaq.sample(kernel, shots_count=int(shots))
    counts = _sample_result_to_counts(sample_result, basis.n_qubits, reverse_bitstrings)
    probabilities = {k: v / float(shots) for k, v in counts.items()}
    resources = estimate_cudaq_resources(sequence, pool, basis)
    return CudaQExecutionResult(counts=counts, probabilities=probabilities, resources=resources)


def estimate_cudaq_resources(
    sequence: list[int] | np.ndarray,
    pool: OperatorPool,
    basis: DeterminantBasis,
) -> dict[str, int | float | dict]:
    """Return paper-style static counts with backend estimate metadata if available."""
    cudaq = _import_cudaq()
    kernel = build_cudaq_kernel(sequence, pool, basis)
    paper_counts: dict[str, int | float | dict] = paper_style_sequence_gate_count(sequence, pool, basis)
    try:
        backend_resources = _resources_to_dict(cudaq.estimate_resources(kernel))
    except Exception as exc:  # pragma: no cover - backend/version dependent
        paper_counts["cudaq_estimate_error"] = str(exc)
    else:
        paper_counts["cudaq_estimate"] = backend_resources
    return paper_counts
