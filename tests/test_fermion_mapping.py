import numpy as np
import pytest


def test_excitation_pauli_terms_match_determinant_rotation():
    pytest.importorskip("pyscf")
    scipy_linalg = pytest.importorskip("scipy.linalg")

    from gqkae.chemistry import make_determinant_basis
    from gqkae.circuits import _bitstring_to_occ_int, apply_excitation_rotation
    from gqkae.fermion_mapping import excitation_pauli_terms, pauli_terms_matrix
    from gqkae.operator_pool import build_h4_uccsd_pool

    basis = make_determinant_basis(4, 2, 2)
    pool = build_h4_uccsd_pool(basis)
    full_indices = [_bitstring_to_occ_int(bit) for bit in basis.bitstrings]
    eye = np.eye(basis.size, dtype=complex)

    for op in (pool[1], pool[9]):  # one single and one double excitation
        terms = excitation_pauli_terms(op, basis.n_qubits)
        assert terms
        assert all(isinstance(term.coefficient, float) for term in terms)
        h_full = pauli_terms_matrix(terms)
        h_restricted = h_full[np.ix_(full_indices, full_indices)]
        assert np.max(np.abs(h_restricted - h_restricted.conj().T)) < 1e-12
        unitary = scipy_linalg.expm(-1j * op.angle * h_restricted)
        for col in range(basis.size):
            expected = apply_excitation_rotation(eye[:, col], op, basis)
            actual = unitary @ eye[:, col]
            assert np.linalg.norm(actual - expected) < 1e-10
