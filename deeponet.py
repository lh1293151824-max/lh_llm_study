"""DeepONet vocabulary output head."""

import math
from typing import Callable

import torch
import torch.nn as nn


class DeepONetOutputHead(nn.Module):
    """Map contextual states and vocabulary embeddings to vocabulary logits.

    Args:
        dim: Transformer hidden and token embedding dimension.
        operator_rank: Shared Branch/Trunk feature dimension.
        norm_eps: RMSNorm epsilon.
        dropout: Branch/Trunk dropout probability.
        norm_layer: Project RMSNorm class or compatible constructor.

    Shapes:
        x: ``[batch_size, seq_len, dim]``.
        token_embedding_weight: ``[vocab_size, dim]``.
        return: ``[batch_size, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        dim: int,
        operator_rank: int,
        norm_eps: float,
        dropout: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self._validate_init_args(
            dim=dim,
            operator_rank=operator_rank,
            norm_eps=norm_eps,
            dropout=dropout,
            norm_layer=norm_layer,
        )

        self.dim = dim
        self.operator_rank = operator_rank

        self.branch_norm = norm_layer(dim, eps=norm_eps)
        self.branch_input = nn.Linear(dim, dim, bias=True)
        self.branch_output = nn.Linear(dim, operator_rank, bias=True)

        self.trunk_norm = norm_layer(dim, eps=norm_eps)
        self.trunk_input = nn.Linear(dim, dim, bias=True)
        self.trunk_output = nn.Linear(dim, operator_rank, bias=True)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _validate_init_args(
        dim: int,
        operator_rank: int,
        norm_eps: float,
        dropout: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        for name, value in {
            "dim": dim,
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
            not isinstance(norm_eps, (int, float))
            or isinstance(norm_eps, bool)
        ):
            raise TypeError("norm_eps must be a number")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and greater than 0")

        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
        ):
            raise TypeError("dropout must be a number")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        if not callable(norm_layer):
            raise TypeError("norm_layer must be callable")

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch_output(
            self.dropout(
                self.activation(
                    self.branch_input(self.branch_norm(x))
                )
            )
        )

    def _trunk(
        self,
        token_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.trunk_output(
            self.dropout(
                self.activation(
                    self.trunk_input(
                        self.trunk_norm(token_embedding_weight)
                    )
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        token_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(
                f"x must have shape [B, S, {self.dim}], "
                f"got {tuple(x.shape)}"
            )
        if (
            token_embedding_weight.ndim != 2
            or token_embedding_weight.size(-1) != self.dim
        ):
            raise ValueError(
                "token_embedding_weight must have shape "
                f"[V, {self.dim}], "
                f"got {tuple(token_embedding_weight.shape)}"
            )
        if x.device != token_embedding_weight.device:
            raise ValueError(
                "x and token_embedding_weight must be on the same device"
            )

        branch_features = self._branch(x)
        trunk_features = self._trunk(token_embedding_weight)
        if branch_features.dtype != trunk_features.dtype:
            trunk_features = trunk_features.to(dtype=branch_features.dtype)

        operator_logits = torch.einsum(
            "bsr,vr->bsv",
            branch_features,
            trunk_features,
        )
        return operator_logits / math.sqrt(self.operator_rank)
