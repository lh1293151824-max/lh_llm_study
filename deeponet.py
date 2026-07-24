"""DeepONet vocabulary output head."""

import math

import torch
import torch.nn as nn


class DeepONetOutputHead(nn.Module):
    """Map contextual states and normalized vocabulary IDs to logits.

    Args:
        dim: Transformer hidden and token embedding dimension.
        vocab_size: Number of vocabulary IDs.
        operator_rank: Shared Branch/Trunk feature dimension.
        dropout: Branch/Trunk dropout probability.

    Shapes:
        x: ``[batch_size, seq_len, dim]``.
        normalized_vocab_ids: ``[vocab_size, 1]``.
        return: ``[batch_size, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        operator_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            dim=dim,
            vocab_size=vocab_size,
            operator_rank=operator_rank,
            dropout=dropout,
        )

        self.dim = dim
        self.vocab_size = vocab_size
        self.operator_rank = operator_rank

        self.branch_input = nn.Linear(dim, dim, bias=True)
        self.branch_output = nn.Linear(dim, operator_rank, bias=True)

        self.trunk_input = nn.Linear(1, dim, bias=True)
        self.trunk_output = nn.Linear(dim, operator_rank, bias=True)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        vocab_ids = torch.arange(
            vocab_size,
            dtype=torch.float32,
        ).unsqueeze(-1)
        vocab_id_mean = vocab_ids.mean()
        vocab_id_std = vocab_ids.std(
            unbiased=False,
        ).clamp_min(
            torch.finfo(vocab_ids.dtype).eps
        )
        normalized_vocab_ids = (
            vocab_ids - vocab_id_mean
        ) / vocab_id_std
        self.register_buffer(
            "normalized_vocab_ids",
            normalized_vocab_ids,
            persistent=False,
        )

    @staticmethod
    def _validate_init_args(
        dim: int,
        vocab_size: int,
        operator_rank: int,
        dropout: float,
    ) -> None:
        for name, value in {
            "dim": dim,
            "vocab_size": vocab_size,
            "operator_rank": operator_rank,
        }.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
        ):
            raise TypeError("dropout must be a number")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch_output(
            self.dropout(
                self.activation(
                    self.branch_input(x)
                )
            )
        )

    def _trunk(self) -> torch.Tensor:
        normalized_vocab_ids = self.normalized_vocab_ids
        if normalized_vocab_ids.dtype != self.trunk_input.weight.dtype:
            normalized_vocab_ids = normalized_vocab_ids.to(
                dtype=self.trunk_input.weight.dtype
            )

        return self.trunk_output(
            self.dropout(
                self.activation(
                    self.trunk_input(normalized_vocab_ids)
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(
                f"x must have shape [B, S, {self.dim}], "
                f"got {tuple(x.shape)}"
            )

        branch_features = self._branch(x)
        trunk_features = self._trunk()
        if branch_features.dtype != trunk_features.dtype:
            trunk_features = trunk_features.to(dtype=branch_features.dtype)

        operator_logits = torch.einsum(
            "bsr,vr->bsv",
            branch_features,
            trunk_features,
        )
        return operator_logits
