from gqkae.config import apply_overrides, load_config


def test_h4_config_load_and_override():
    cfg = load_config("configs/h4_mvp.yaml")
    assert cfg.molecule.name == "H4"
    assert cfg.operator_pool.sequence_length == 20
    apply_overrides(cfg, ["training.iterations=2", "qsci.shots=50"])
    assert cfg.training.iterations == 2
    assert cfg.qsci.shots == 50
