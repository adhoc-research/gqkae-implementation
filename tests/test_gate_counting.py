from collections import Counter

import pytest


def test_paper_style_pauli_evolution_count_matches_cited_formula():
    from gqkae.gate_counting import paper_style_pauli_evolution_gate_count

    assert paper_style_pauli_evolution_gate_count("IIII") == Counter(
        {"cx": 0, "h": 0, "s": 0, "sdg": 0, "rz": 0, "two_qubit": 0, "total": 0}
    )
    assert paper_style_pauli_evolution_gate_count("XZYI") == Counter(
        {"cx": 4, "h": 4, "s": 1, "sdg": 1, "rz": 1, "two_qubit": 4, "total": 11}
    )


def test_paper_style_sequence_count_h4_representative_tokens():
    pytest.importorskip("pyscf")

    from gqkae.chemistry import make_determinant_basis
    from gqkae.config import OperatorPoolConfig
    from gqkae.gate_counting import paper_style_sequence_gate_count
    from gqkae.operator_pool import build_h4_uccsd_pool

    basis = make_determinant_basis(4, 2, 2)
    pool = build_h4_uccsd_pool(basis, OperatorPoolConfig(sequence_length=20, include_noop=True))

    assert paper_style_sequence_gate_count([0] * 20, pool, basis) == {
        "x": 4,
        "h": 0,
        "s": 0,
        "sdg": 0,
        "rz": 0,
        "cx": 0,
        "two_qubit": 0,
        "total": 4,
    }
    assert paper_style_sequence_gate_count([1, 9], pool, basis) == {
        "x": 4,
        "h": 72,
        "s": 18,
        "sdg": 18,
        "rz": 10,
        "cx": 56,
        "two_qubit": 56,
        "total": 178,
    }
