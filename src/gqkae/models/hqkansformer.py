"""Autoregressive HQKANsformer policy for operator-token generation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .qkan import HQKAN


class HQKANTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(config.d_model)
        self.hqkan = HQKAN(
            d_model=config.d_model,
            latent_dim=config.latent_dim,
            qkan_layers=config.qkan_layers,
            terms=config.daruan_terms,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        y = self.ln_attn(x)
        attn_out, _ = self.attn(y, y, y, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.hqkan(self.ln_ff(x)))
        return x


class HQKANsformerPolicy(nn.Module):
    """Decoder-only policy with HQKAN feed-forward replacements.

    Token IDs `0..vocab_size-1` are operator tokens. The dedicated BOS token is
    `vocab_size` and is used only as an input embedding, never as an output action.
    """

    def __init__(self, vocab_size: int, sequence_length: int, config: ModelConfig):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bos_token = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size + 1, config.d_model)
        self.position_embedding = nn.Embedding(sequence_length, config.d_model)
        self.blocks = nn.ModuleList([HQKANTransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Return logits for next operator at each input position.

        ``input_tokens`` has shape ``(batch, length)`` and may include the BOS token.
        """
        batch, length = input_tokens.shape
        if length > self.sequence_length:
            raise ValueError(f"length {length} exceeds configured {self.sequence_length}")
        positions = torch.arange(length, device=input_tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.full((length, length), float("-inf"), device=input_tokens.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.output(self.ln_f(x))

    def input_from_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        bos = torch.full(
            (sequences.shape[0], 1),
            self.bos_token,
            dtype=torch.long,
            device=sequences.device,
        )
        return torch.cat([bos, sequences[:, :-1]], dim=1)

    def log_probs_for_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        inputs = self.input_from_sequences(sequences)
        logits = self(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, sequences.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample operator sequences and return (tokens, old_log_probs)."""
        if device is None:
            device = next(self.parameters()).device
        sequences: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        prefix = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=device)
        generated = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        for _ in range(self.sequence_length):
            logits = self(prefix)[:, -1, :] / max(float(temperature), 1e-8)
            if repetition_penalty and repetition_penalty != 1.0 and generated.numel() > 0:
                for row in range(batch_size):
                    used = generated[row].unique()
                    logits[row, used] = logits[row, used] / float(repetition_penalty)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(probs.gather(-1, token).clamp_min(1e-30)).squeeze(-1)
            sequences.append(token.squeeze(-1))
            old_log_probs.append(log_prob)
            generated = torch.cat([generated, token], dim=1)
            prefix = torch.cat([prefix, token], dim=1)

        return torch.stack(sequences, dim=1), torch.stack(old_log_probs, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_memory_mb(model: nn.Module) -> float:
    return float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2))
