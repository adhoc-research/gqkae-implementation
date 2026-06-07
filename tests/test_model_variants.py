import pytest


def test_gpt_policy_shape_and_grad():
    torch = pytest.importorskip("torch")
    from gqkae.config import ModelConfig
    from gqkae.models.gpt_policy import GPTPolicy

    cfg = ModelConfig(variant="gpt", d_model=16, n_layers=1, n_heads=4, d_ff=32)
    model = GPTPolicy(vocab_size=7, sequence_length=5, config=cfg)
    tokens = torch.full((2, 5), model.bos_token, dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 5, 7)
    loss = logits.square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_runner_model_variant_config_instantiation():
    pytest.importorskip("torch")
    from gqkae.config import ModelConfig
    from gqkae.models.gpt_policy import GPTPolicy
    from gqkae.models.hqkansformer import HQKANsformerPolicy

    hqkan = HQKANsformerPolicy(7, 5, ModelConfig(variant="hqkan", d_model=16, n_layers=1, n_heads=4, latent_dim=4))
    gpt = GPTPolicy(7, 5, ModelConfig(variant="gpt", d_model=16, n_layers=1, n_heads=4, d_ff=32))
    assert hqkan.vocab_size == gpt.vocab_size == 7
