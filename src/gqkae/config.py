"""Configuration objects for the H4 GQKAE MVP."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "h4_mvp"
    seed: int = 7
    output_dir: str = "runs/h4_mvp"


@dataclass(slots=True)
class MoleculeConfig:
    name: str = "H4"
    bond_length_angstrom: float = 0.88
    basis: str = "6-31g"
    active_electrons: int = 4
    active_orbitals: int = 4
    geometry: str = "linear_chain"


@dataclass(slots=True)
class OperatorPoolConfig:
    sequence_length: int = 20
    excitation_angle: float = 0.7853981633974483
    include_noop: bool = True
    spec: str = "excitation"
    ccsd_threshold: float = 1e-6
    remove_z_ladder: bool = False
    only_use_first_pauli: bool = False
    dedupe_pauli_words: bool = True


@dataclass(slots=True)
class QSCIConfig:
    shots: int = 1_000
    dmax: int = 100
    add_hf_det: bool = True
    backend: str = "determinant"
    cudaq_target: str | None = None
    cudaq_option: str | None = None
    cudaq_reverse_bitstrings: bool = False


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    latent_dim: int = 12
    qkan_layers: int = 1
    daruan_terms: int = 4
    dropout: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    iterations: int = 10
    batch_circuits: int = 4
    policy_updates: int = 2
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    grpo_clip: float = 0.2
    repetition_penalty: float = 1.2
    temperature: float = 1.0
    device: str = "cpu"


@dataclass(slots=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    molecule: MoleculeConfig = field(default_factory=MoleculeConfig)
    operator_pool: OperatorPoolConfig = field(default_factory=OperatorPoolConfig)
    qsci: QSCIConfig = field(default_factory=QSCIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _construct(cls: type, data: Mapping[str, Any] | None):
    if data is None:
        return cls()
    kwargs = {}
    for f in fields(cls):
        value = data.get(f.name, getattr(cls(), f.name))
        if is_dataclass(f.type):
            value = _construct(f.type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        experiment=_construct(ExperimentConfig, raw.get("experiment")),
        molecule=_construct(MoleculeConfig, raw.get("molecule")),
        operator_pool=_construct(OperatorPoolConfig, raw.get("operator_pool")),
        qsci=_construct(QSCIConfig, raw.get("qsci")),
        model=_construct(ModelConfig, raw.get("model")),
        training=_construct(TrainingConfig, raw.get("training")),
    )


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: Config, overrides: list[str] | None) -> Config:
    """Apply CLI overrides like ``training.iterations=2`` in place."""
    if not overrides:
        return config
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.field=value, got {item!r}")
        key, value = item.split("=", 1)
        section, field = key.split(".", 1)
        target = getattr(config, section)
        if not hasattr(target, field):
            raise ValueError(f"unknown config field {key!r}")
        setattr(target, field, _coerce_scalar(value))
    return config


def as_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return {f.name: as_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
