"""DeepONet 语言模型输出头设计原型。

DESIGN PROTOTYPE ONLY
=====================

这个文件用于在正式修改 ``k_model.py``、``train.py`` 和 ``config.py`` 之前，
集中展示并讨论 DeepONet 输出头的算法、参数、张量维度和权重共享方式。
当前文件没有接入 Transformer，也不会被现有训练或生成流程导入。

核心符号
--------

    B: batch size
    S: sequence length
    V: vocabulary size
    D: Transformer hidden size / token embedding size
    Hb: BranchNet hidden size
    Ht: TrunkNet hidden size
    R: operator rank, 即 Branch/Trunk 的公共特征维度

数据流
------

    Transformer hidden states x [B, S, D]
        -> BranchNet
        -> branch_features [B, S, R]

    Transformer.tok_embeddings.weight [V, D]
        -> TrunkNet
        -> trunk_features [V, R]

    einsum("bsr,vr->bsv")
        -> operator_logits [B, S, V]

三种输出模式
------------

    linear:
        logits = linear_logits

    deeponet:
        logits = operator_logits

    hybrid:
        logits = (1 - alpha) * linear_logits + alpha * operator_logits

正式接入时，所有超参数必须从项目配置文件传入。本文件不提供用于正式训练的
硬编码超参数。
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

import torch
import torch.nn as nn


class DeepONetConfigProtocol(Protocol):
    """正式配置对象需要提供的字段，仅用于类型和设计说明。

    建议在项目配置文件中加入的字段：

    output_head_type: "linear" | "deeponet" | "hybrid"
    operator_rank: R
    operator_branch_hidden_dim: Hb
    operator_trunk_hidden_dim: Ht
    operator_activation: "silu" | "gelu" | "relu"
    operator_dropout: Branch/Trunk 内部 dropout
    operator_alpha_init: hybrid 模式下 DeepONet 的初始占比
    operator_alpha_learnable: alpha 是否参与训练
    operator_cache_trunk_eval: 推理时是否缓存 [V, R]

    ``dim`` 和 ``vocab_size`` 复用现有 ModelConfig 字段。
    """

    dim: int
    vocab_size: int
    output_head_type: str
    operator_rank: int
    operator_branch_hidden_dim: int
    operator_trunk_hidden_dim: int
    operator_activation: str
    operator_dropout: float
    operator_alpha_init: float
    operator_alpha_learnable: bool
    operator_cache_trunk_eval: bool


def validate_deeponet_config(config: DeepONetConfigProtocol) -> None:
    """在创建模块前检查配置，避免错误延迟到矩阵乘法阶段。"""

    valid_modes = {"linear", "deeponet", "hybrid"}
    if config.output_head_type not in valid_modes:
        raise ValueError(
            f"output_head_type must be one of {sorted(valid_modes)}, "
            f"got {config.output_head_type!r}"
        )

    positive_integer_fields = {
        "dim": config.dim,
        "vocab_size": config.vocab_size,
        "operator_rank": config.operator_rank,
        "operator_branch_hidden_dim": config.operator_branch_hidden_dim,
        "operator_trunk_hidden_dim": config.operator_trunk_hidden_dim,
    }
    for name, value in positive_integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if config.operator_activation not in {"silu", "gelu", "relu"}:
        raise ValueError(
            "operator_activation must be one of: silu, gelu, relu"
        )

    if not 0.0 <= config.operator_dropout < 1.0:
        raise ValueError("operator_dropout must be in [0, 1)")

    if not 0.0 <= config.operator_alpha_init <= 1.0:
        raise ValueError("operator_alpha_init must be in [0, 1]")

    # Sigmoid 的输出严格位于 (0, 1)，无法用有限参数精确表示端点。
    if config.operator_alpha_learnable and not (
        0.0 < config.operator_alpha_init < 1.0
    ):
        raise ValueError(
            "learnable operator_alpha_init must be strictly between 0 and 1"
        )


def build_activation(name: str) -> nn.Module:
    """根据配置创建激活函数。"""

    activations = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name!r}") from exc


class BranchNet(nn.Module):
    """把每个上下文 token 的隐藏状态映射到算子特征空间。

    输入:
        x: [B, S, D]

    输出:
        branch_features: [B, S, R]

    权重:
        input_projection.weight:  [Hb, D]
        input_projection.bias:    [Hb]
        output_projection.weight: [R, Hb]
        output_projection.bias:   [R]
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        operator_rank: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.operator_rank = operator_rank

        self.input_projection = nn.Linear(model_dim, hidden_dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, operator_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.model_dim:
            raise ValueError(
                f"BranchNet expects [B, S, {self.model_dim}], "
                f"got {tuple(x.shape)}"
            )

        branch_features = self.output_projection(
            self.dropout(self.activation(self.input_projection(x)))
        )
        return branch_features


class TrunkNet(nn.Module):
    """把词表中每个 token 的共享 embedding 映射到算子特征空间。

    输入:
        token_embedding_weight: [V, D]

    输出:
        trunk_features: [V, R]

    权重:
        input_projection.weight:  [Ht, D]
        input_projection.bias:    [Ht]
        output_projection.weight: [R, Ht]
        output_projection.bias:   [R]

    注意：TrunkNet 自己的权重不与 BranchNet 共享。共享的是传入的
    ``Transformer.tok_embeddings.weight``。
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        operator_rank: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.operator_rank = operator_rank

        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, operator_rank)

    def forward(self, token_embedding_weight: torch.Tensor) -> torch.Tensor:
        if (
            token_embedding_weight.ndim != 2
            or token_embedding_weight.size(-1) != self.embedding_dim
        ):
            raise ValueError(
                f"TrunkNet expects [V, {self.embedding_dim}], "
                f"got {tuple(token_embedding_weight.shape)}"
            )

        trunk_features = self.output_projection(
            self.dropout(
                self.activation(self.input_projection(token_embedding_weight))
            )
        )
        return trunk_features


class DeepONetOutputHead(nn.Module):
    """三模式语言模型输出头的设计参考实现。

    本模块不保存 ``tok_embeddings`` 或 ``linear_head`` 的模块引用，避免在
    Transformer 内产生重复的子模块注册。正式接入时由 Transformer.forward
    显式传入：

        token_embedding_weight = self.tok_embeddings.weight
        linear_logits = self.output(x)  # deeponet 模式可传 None

    这种写法仍然是权重共享，因为 TrunkNet 接收到的 Tensor 就是原始
    ``tok_embeddings.weight`` Parameter，而不是 clone 或 detach 后的副本。
    """

    def __init__(self, config: DeepONetConfigProtocol) -> None:
        super().__init__()
        validate_deeponet_config(config)

        self.output_head_type = config.output_head_type
        self.model_dim = config.dim
        self.vocab_size = config.vocab_size
        self.operator_rank = config.operator_rank
        self.alpha_learnable = config.operator_alpha_learnable
        self.cache_trunk_eval = config.operator_cache_trunk_eval

        self.branch_net = BranchNet(
            model_dim=config.dim,
            hidden_dim=config.operator_branch_hidden_dim,
            operator_rank=config.operator_rank,
            activation=config.operator_activation,
            dropout=config.operator_dropout,
        )
        self.trunk_net = TrunkNet(
            embedding_dim=config.dim,
            hidden_dim=config.operator_trunk_hidden_dim,
            operator_rank=config.operator_rank,
            activation=config.operator_activation,
            dropout=config.operator_dropout,
        )

        alpha_init = float(config.operator_alpha_init)
        if self.alpha_learnable:
            alpha_logit_init = math.log(alpha_init / (1.0 - alpha_init))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))
            self.register_buffer("fixed_alpha", None, persistent=False)
        else:
            self.register_parameter("alpha_logit", None)
            self.register_buffer(
                "fixed_alpha",
                torch.tensor(alpha_init),
                persistent=True,
            )

        # 仅为当前进程的推理加速缓存，不写入 checkpoint。
        self._cached_trunk_features: Optional[torch.Tensor] = None
        self._cached_embedding_version: Optional[int] = None
        self._cached_embedding_data_ptr: Optional[int] = None
        self._cached_trunk_parameter_versions: Optional[tuple[int, ...]] = None

    def get_alpha(self) -> torch.Tensor:
        """返回标量 alpha，并保证其位于 [0, 1]。"""

        if self.alpha_learnable:
            return torch.sigmoid(self.alpha_logit)
        return self.fixed_alpha

    def clear_trunk_cache(self) -> None:
        """清除推理缓存；加载权重、切换训练时都应调用。"""

        self._cached_trunk_features = None
        self._cached_embedding_version = None
        self._cached_embedding_data_ptr = None
        self._cached_trunk_parameter_versions = None

    def train(self, mode: bool = True) -> "DeepONetOutputHead":
        """训练模式不能复用旧 Trunk 特征。"""

        if mode:
            self.clear_trunk_cache()
        return super().train(mode)

    def _trunk_parameter_versions(self) -> tuple[int, ...]:
        """记录 Trunk 参数版本，用于检测原地参数更新。

        ``Tensor._version`` 属于 PyTorch 内部机制。正式实现前需要决定：
        继续使用版本签名自动失效，还是由 Transformer 在加载权重、切换设备、
        开始/结束生成时显式调用 ``clear_trunk_cache``。
        """

        return tuple(parameter._version for parameter in self.trunk_net.parameters())

    def _get_trunk_features(
        self,
        token_embedding_weight: torch.Tensor,
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        """训练时实时计算，推理时按配置安全复用缓存。"""

        cache_allowed = (
            self.cache_trunk_eval
            and not self.training
            and not torch.is_grad_enabled()
        )
        if not cache_allowed:
            return self.trunk_net(token_embedding_weight)

        embedding_version = token_embedding_weight._version
        embedding_data_ptr = token_embedding_weight.data_ptr()
        trunk_versions = self._trunk_parameter_versions()
        cache_is_valid = (
            self._cached_trunk_features is not None
            and self._cached_embedding_version == embedding_version
            and self._cached_embedding_data_ptr == embedding_data_ptr
            and self._cached_trunk_parameter_versions == trunk_versions
            and self._cached_trunk_features.device
            == token_embedding_weight.device
            and self._cached_trunk_features.dtype == expected_dtype
        )
        if not cache_is_valid:
            self._cached_trunk_features = self.trunk_net(
                token_embedding_weight
            ).detach()
            self._cached_embedding_version = embedding_version
            self._cached_embedding_data_ptr = embedding_data_ptr
            self._cached_trunk_parameter_versions = trunk_versions

        return self._cached_trunk_features

    def compute_operator_logits(
        self,
        x: torch.Tensor,
        token_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """计算 [B, S, V] 的算子映射 logits。"""

        if token_embedding_weight.size(0) != self.vocab_size:
            raise ValueError(
                "token embedding vocabulary dimension does not match config"
            )
        if x.device != token_embedding_weight.device:
            raise ValueError("x and token embedding weight must be on one device")

        branch_features = self.branch_net(x)  # [B, S, R]
        trunk_features = self._get_trunk_features(
            token_embedding_weight,
            expected_dtype=branch_features.dtype,
        )  # [V, R]

        if branch_features.dtype != trunk_features.dtype:
            trunk_features = trunk_features.to(dtype=branch_features.dtype)

        operator_logits = torch.einsum(
            "bsr,vr->bsv",
            branch_features,
            trunk_features,
        )
        return operator_logits / math.sqrt(self.operator_rank)

    def forward(
        self,
        x: torch.Tensor,
        token_embedding_weight: torch.Tensor,
        linear_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """根据配置返回最终 [B, S, V] logits。"""

        if self.output_head_type == "linear":
            if linear_logits is None:
                raise ValueError("linear mode requires linear_logits")
            return linear_logits

        operator_logits = self.compute_operator_logits(
            x=x,
            token_embedding_weight=token_embedding_weight,
        )

        if self.output_head_type == "deeponet":
            return operator_logits

        if linear_logits is None:
            raise ValueError("hybrid mode requires linear_logits")
        if linear_logits.shape != operator_logits.shape:
            raise ValueError(
                "linear and operator logits must have identical shapes"
            )

        alpha = self.get_alpha().to(
            device=operator_logits.device,
            dtype=operator_logits.dtype,
        )
        return (1.0 - alpha) * linear_logits + alpha * operator_logits


# ---------------------------------------------------------------------------
# Transformer 中的预期接入伪代码（当前不要执行或直接复制进 k_model.py）
# ---------------------------------------------------------------------------
#
# class Transformer(...):
#     def __init__(self, config):
#         ...
#         self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
#         self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
#         self.output.weight = self.tok_embeddings.weight
#
#         if config.output_head_type in {"deeponet", "hybrid"}:
#             self.operator_output = DeepONetOutputHead(config)
#         else:
#             self.operator_output = None
#
#     def compute_logits(self, x):
#         # 保留现有 Linear 权重绑定。纯 deeponet 模式不必执行 Linear GEMM。
#         linear_logits = None
#         if self.config.output_head_type in {"linear", "hybrid"}:
#             linear_logits = self.output(x)
#
#         if self.operator_output is None:
#             return linear_logits
#
#         return self.operator_output(
#             x=x,
#             token_embedding_weight=self.tok_embeddings.weight,
#             linear_logits=linear_logits,
#         )
#
#     def forward(self, idx, targets=None, attention_mask=None):
#         ...
#         x = self.norm(x)
#         logits = self.compute_logits(x)
#         ...
#         return {"logits": logits, "last_loss": loss}
#
# 现有 generate() 不需要修改。它继续读取：
#     outputs = self(idx_cond, attention_mask=mask_cond)
#     logits = outputs["logits"][:, -1, :]


# ---------------------------------------------------------------------------
# 权重共享与梯度路径
# ---------------------------------------------------------------------------
#
# 同一个 Parameter：Transformer.tok_embeddings.weight [V, D]
#
# 1. 输入路径：
#     token ids -> tok_embeddings -> Transformer -> loss
#
# 2. Linear 输出路径：
#     hidden x -> output(weight tied to tok_embeddings.weight) -> loss
#
# 3. Trunk 输出路径：
#     tok_embeddings.weight -> TrunkNet -> operator logits -> loss
#
# 三条路径的梯度累加到同一个 tok_embeddings.weight.grad。这里不能对传给
# TrunkNet 的权重调用 clone().detach()，否则会切断 Trunk 路径对 embedding
# 的梯度。BranchNet 和 TrunkNet 的内部 Linear 权重彼此独立。


# ---------------------------------------------------------------------------
# Checkpoint 兼容设计
# ---------------------------------------------------------------------------
#
# 1. 旧配置没有 output_head_type 时，正式配置构建函数默认补为 "linear"。
# 2. 旧 checkpoint + linear：行为与当前模型保持一致。
# 3. 旧 checkpoint + hybrid：加载旧 Linear/Embedding，Branch/Trunk 为新增参数，
#    alpha 使用较小的配置初值，使初始行为接近 Linear 基线。
# 4. 旧 checkpoint + deeponet：输出头无法与旧 Linear 完全等价，需要训练新增
#    Branch/Trunk 参数后才能获得可靠输出。
# 5. load_state_dict() 完成后必须调用 clear_trunk_cache()。


# ---------------------------------------------------------------------------
# 仍需和用户确定的问题
# ---------------------------------------------------------------------------
#
# 1. BranchNet/TrunkNet 是否需要 RMSNorm 或 LayerNorm。
# 2. operator_logits / sqrt(R) 是否足以匹配 Linear logits 的数值尺度，还是需要
#    增加可学习 temperature/scale。该参数如采用，也必须进入配置文件。
# 3. alpha 是全模型单标量、每层/每 token 标量，还是每个词表 token 一个值。
#    当前原型采用最稳定、最易解释的全模型单标量。
# 4. TrunkNet 是否保留 bias。当前原型保留两个 Linear 的 bias。
# 5. 是否在纯 deeponet 模式继续保留 self.output 以方便旧 checkpoint 加载。
# 6. 推理缓存采用自动版本检测还是显式生成会话生命周期管理。
# 7. 是否需要给 Branch/Trunk 输出做 L2 normalization。做归一化会把内积变成
#    余弦相似度，并明显改变 logits 尺度和模型表达方式。
# 8. 首轮消融建议比较 linear、deeponet、hybrid，并分别比较多个 operator_rank、
#    固定/可学习 alpha、启用/禁用 Trunk 缓存的精度和速度。
