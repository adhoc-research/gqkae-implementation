import pytest


def test_h4_pool_and_cudaq_source():
    pytest.importorskip("pyscf")
    from gqkae.chemistry import make_determinant_basis
    from gqkae.config import OperatorPoolConfig
    from gqkae.circuits import sequence_to_cudaq_source
    from gqkae.operator_pool import build_h4_uccsd_pool

    basis = make_determinant_basis(4, 2, 2)
    pool = build_h4_uccsd_pool(basis, OperatorPoolConfig(sequence_length=20, include_noop=True))
    assert pool.sequence_length == 20
    assert pool.vocab_size == 27
    source = sequence_to_cudaq_source([0, 1, 9], pool, basis)
    assert "@cudaq.kernel" in source.source
    assert "cudaq.qvector(8)" in source.source
    assert source.gate_count == {
        "x": 4,
        "h": 72,
        "s": 18,
        "sdg": 18,
        "rz": 10,
        "cx": 56,
        "two_qubit": 56,
        "total": 178,
    }
    assert "sdg(q[" in source.source
    assert "s(q[" in source.source
    assert "ry(" not in source.source
