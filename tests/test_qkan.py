import pytest


def test_hqkan_shape_and_grad():
    torch = pytest.importorskip("torch")
    from gqkae.models.qkan import HQKAN

    module = HQKAN(d_model=16, latent_dim=4, qkan_layers=1, terms=3)
    x = torch.randn(2, 5, 16)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert any(p.grad is not None for p in module.parameters())
