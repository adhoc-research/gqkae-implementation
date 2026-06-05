"""Quantum-inspired KAN modules for the HQKANsformer.

DARUAN is implemented as a compact trainable Fourier/data-reuploading activation. This
keeps the MVP differentiable and parameter-efficient while matching the paper's
encoder -> QKAN latent processor -> decoder structure.
"""

from __future__ import annotations

import torch
from torch import nn


class DARUANEdgeBank(nn.Module):
    """Edge-wise scalar nonlinearities for one QKAN layer.

    For each edge i -> j, the learnable univariate response is a sum of reuploaded
    sinusoidal features: sum_k a cos(w_k x + b) + c sin(w_k x + d). This is the
    classical differentiable analogue of the single-qubit data re-uploading activation.
    """

    def __init__(self, in_features: int, out_features: int, terms: int = 4):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.terms = int(terms)
        shape = (out_features, in_features, terms)
        frequencies = torch.arange(1, terms + 1, dtype=torch.float32).view(1, 1, terms)
        self.log_freq_scale = nn.Parameter(torch.zeros(shape))
        self.register_buffer("base_frequencies", frequencies)
        self.cos_weight = nn.Parameter(torch.randn(shape) * 0.02)
        self.sin_weight = nn.Parameter(torch.randn(shape) * 0.02)
        self.cos_phase = nn.Parameter(torch.zeros(shape))
        self.sin_phase = nn.Parameter(torch.zeros(shape))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        x_edge = x.unsqueeze(-2).unsqueeze(-1)  # (..., 1, in, 1)
        freq = self.base_frequencies * torch.exp(self.log_freq_scale)
        values = self.cos_weight * torch.cos(freq * x_edge + self.cos_phase)
        values = values + self.sin_weight * torch.sin(freq * x_edge + self.sin_phase)
        return values.sum(dim=(-1, -2)) + self.bias


class QKANLayer(nn.Module):
    """A QKAN layer with DARUAN edge functions and edge summation."""

    def __init__(self, width: int, terms: int = 4):
        super().__init__()
        self.width = int(width)
        self.edges = DARUANEdgeBank(width, width, terms=terms)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.edges(x)


class QKAN(nn.Module):
    """Stacked latent-space QKAN processor."""

    def __init__(self, width: int, layers: int = 1, terms: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([QKANLayer(width, terms=terms) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class HQKAN(nn.Module):
    """Encoder -> QKAN latent processor -> decoder replacement for transformer FFNs."""

    def __init__(self, d_model: int, latent_dim: int = 12, qkan_layers: int = 1, terms: int = 4):
        super().__init__()
        self.encoder = nn.Linear(d_model, latent_dim)
        self.processor = QKAN(latent_dim, layers=qkan_layers, terms=terms)
        self.decoder = nn.Linear(latent_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.processor(z)
        return self.decoder(z)
