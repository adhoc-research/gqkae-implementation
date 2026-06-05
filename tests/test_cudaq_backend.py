import pytest


def test_cudaq_backend_hf_sampling_if_available():
    cudaq = pytest.importorskip("cudaq")
    pytest.importorskip("pyscf")

    from gqkae.chemistry import make_determinant_basis
    from gqkae.cudaq_backend import sample_cudaq_sequence
    from gqkae.operator_pool import build_h4_uccsd_pool

    basis = make_determinant_basis(4, 2, 2)
    pool = build_h4_uccsd_pool(basis)
    result = sample_cudaq_sequence(
        [0] * pool.sequence_length,
        pool,
        basis,
        shots=20,
        target="qpp-cpu" if hasattr(cudaq, "set_target") else None,
        seed=123,
    )
    assert sum(result.counts.values()) == 20
    assert basis.hf_bitstring in result.counts
    assert result.resources
