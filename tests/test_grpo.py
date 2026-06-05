import pytest


def test_grpo_loss_is_finite():
    torch = pytest.importorskip("torch")
    from gqkae.training.grpo import grpo_loss, normalize_advantages

    new = torch.zeros(3, 4, requires_grad=True)
    old = torch.zeros(3, 4)
    rewards = torch.tensor([1.0, 2.0, 4.0])
    adv = normalize_advantages(rewards)
    loss = grpo_loss(new, old, adv, clip_epsilon=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert new.grad is not None
