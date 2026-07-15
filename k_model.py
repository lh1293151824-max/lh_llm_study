import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class ModelConfig(PretrainedConfig):
    model_type = "tiny-k"

    def __init__(
        self,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        vocab_size: int = 6144,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight * norm_x.type_as(x)


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs.shape == (x.shape[1], x.shape[-1])
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 将freqs_cis调整为新的形状，并返回
    return freqs.view(shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    q_dtype = q.dtype
    k_dtype = k.dtype

    q = q.float().reshape(q.shape[0], q.shape[1], q.shape[2], -1, 2)
    k = k.float().reshape(k.shape[0], k.shape[1], k.shape[2], -1, 2)

    q_r, q_i = q.unbind(-1)
    k_r, k_i = k.unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, q_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, q_i)

    q_out_r = q_r * freqs_cos - q_i * freqs_sin
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos

    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)
    return q_out.to(dtype=q_dtype), k_out.to(dtype=k_dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(
        batch_size,
        seq_len,
        n_kv_heads,
        n_rep,
        head_dim,
    )
    return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(
        self,
        dim_embedding=256,
        n_heads=8,
        n_kv_heads=None,
        dropout=0.0,
    ):
        super().__init__()

        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert dim_embedding % n_heads == 0, "dim_embedding must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim_embedding = dim_embedding
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim_embedding // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim_embedding, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim_embedding, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim_embedding, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim_embedding, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin, attention_mask=None):
        if x.dim() != 3:
            raise ValueError("x must have shape [batch_size, seq_len, dim]")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

        batch_size, seq_len, dim_embedding = x.shape
        if dim_embedding != self.dim_embedding:
            raise ValueError(
                f"Expected x.size(-1)={self.dim_embedding}, "
                f"got {dim_embedding}"
            )

        expected_freqs_shape = (seq_len, self.head_dim // 2)
        if freqs_cos.shape != expected_freqs_shape:
            raise ValueError(
                f"freqs_cos must have shape {expected_freqs_shape}, "
                f"got {tuple(freqs_cos.shape)}"
            )
        if freqs_sin.shape != expected_freqs_shape:
            raise ValueError(
                f"freqs_sin must have shape {expected_freqs_shape}, "
                f"got {tuple(freqs_sin.shape)}"
            )

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.shape != (batch_size, seq_len):
                    raise ValueError(
                        "A 2D attention_mask must have shape "
                        f"[{batch_size}, {seq_len}]"
                    )
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                expected_mask_shape = (batch_size, 1, 1, seq_len)
                if attention_mask.shape != expected_mask_shape:
                    raise ValueError(
                        "A 4D attention_mask must have shape "
                        f"{expected_mask_shape}"
                    )
            else:
                raise ValueError("attention_mask must be either 2D or 4D")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        if attention_mask is not None:
            mask = mask[None, None, :, :] & attention_mask
            empty_rows = ~mask.any(dim=-1, keepdim=True)
            diagonal_mask = torch.eye(
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )[None, None, :, :]
            mask = mask | (empty_rows & diagonal_mask)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_embedding)
        return self.resid_dropout(self.wo(out))


class MLP(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        hidden_dim: int = None,
        multiple_of: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim_embedding
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim_embedding = dim_embedding
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim_embedding, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim_embedding, bias=False)
        self.w3 = nn.Linear(dim_embedding, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding=256,
        n_heads=8,
        n_kv_heads=None,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        hidden_dim=None,
        multiple_of: int = 64,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim_embedding, eps=norm_eps)
        self.attention = Attention(
            dim_embedding=dim_embedding,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(dim_embedding, eps=norm_eps)
        self.feed_forward = MLP(
            dim_embedding=dim_embedding,
            hidden_dim=hidden_dim,
            multiple_of=multiple_of,
            dropout=dropout,
        )

    def forward(self, x, freqs_cos, freqs_sin, attention_mask=None):
        x = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            attention_mask=attention_mask,
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(PreTrainedModel):
    config_class = ModelConfig
    _tied_weights_keys = {"output.weight": "tok_embeddings.weight"}

    def __init__(
        self,
        config=None,
        vocab_size=256,
        max_seq_len=3,
        dim_embedding=256,
        n_heads=8,
        n_layers=4,
    ):
        if config is None:
            config = ModelConfig(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                dim=dim_embedding,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                n_layers=n_layers,
            )

        super().__init__(config)
        self.config = config
        self.OUT = {}
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.dim = config.dim
        self.dim_embedding = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.norm_eps = config.norm_eps
        self.dropout = config.dropout
        self.hidden_dim = config.hidden_dim
        self.multiple_of = config.multiple_of

        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.head_dim = self.dim // self.n_heads
        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even"

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim_embedding=self.dim,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    norm_eps=self.norm_eps,
                    dropout=self.dropout,
                    hidden_dim=self.hidden_dim,
                    multiple_of=self.multiple_of,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.dim, eps=self.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.n_layers),
                )

        self.output.weight = self.tok_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None):
        if idx.dim() != 2:
            raise ValueError("idx must have shape [batch_size, seq_len]")
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("idx must use torch.int32 or torch.int64")

        batch_size, seq_len = idx.shape
        if seq_len == 0:
            raise ValueError("idx must contain at least one token")
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence length exceeds max_seq_len")

        if targets is not None:
            if targets.dim() != 2 or targets.shape != idx.shape:
                raise ValueError("targets must have the same 2D shape as idx")
            if targets.dtype != torch.int64:
                raise TypeError("targets must use torch.int64")
            if targets.device != idx.device:
                raise ValueError("targets and idx must be on the same device")

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != idx.shape:
                raise ValueError(
                    "attention_mask must have the same 2D shape as idx"
                )
            attention_mask = attention_mask.to(
                device=idx.device,
                dtype=torch.bool,
            )

        x = self.embedding_dropout(self.tok_embeddings(idx))
        freqs_cos = self.freqs_cos[:seq_len].to(x.device)
        freqs_sin = self.freqs_sin[:seq_len].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, attention_mask=attention_mask)

        x = self.norm(x)

        if targets is None:

            logits = self.output(x[:, [-1], :])
            self.last_loss = None
        else:
            logits = self.output(x)

            batch_size, seq_len, vocab_size = logits.shape
            loss_targets = targets
            if attention_mask is not None:
                loss_targets = targets.masked_fill(
                    ~attention_mask.bool(),
                    -100,
                )

            self.last_loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                loss_targets.reshape(batch_size * seq_len),
                ignore_index=-100,
                reduction="none",
            )
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(
        self,
        idx,
        stop_id=None,
        max_new_tokens=256,
        temperature=1.0,
        top_k=None,
        attention_mask=None,
        pad_token_id=None,
    ):
        if idx.dim() != 2:
            raise ValueError("idx must have shape [batch_size, seq_len]")
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("idx must use torch.int32 or torch.int64")
        if idx.size(1) == 0:
            raise ValueError("idx must contain at least one token")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")

        if pad_token_id is None:
            pad_token_id = (
                self.config.pad_token_id
                if self.config.pad_token_id is not None
                else 0
            )
        if not isinstance(pad_token_id, int) or not 0 <= pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a valid vocabulary index")
        if stop_id is not None and (
            not isinstance(stop_id, int) or not 0 <= stop_id < self.vocab_size
        ):
            raise ValueError("stop_id must be a valid vocabulary index or None")

        if attention_mask is None:
            if idx.size(0) == 1:
                attention_mask = torch.ones_like(idx, dtype=torch.bool)
            else:
                raise ValueError(
                    "attention_mask is required when batch size is greater than 1"
                )
        else:
            if attention_mask.dim() != 2 or attention_mask.shape != idx.shape:
                raise ValueError(
                    "attention_mask must have the same 2D shape as idx"
                )
            attention_mask = attention_mask.to(
                device=idx.device,
                dtype=torch.bool,
            )

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        start_index = idx.size(1)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            mask_cond = attention_mask[:, -self.max_seq_len :]

            outputs = self(idx_cond, attention_mask=mask_cond)
            logits = outputs["logits"][:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    top_values, _ = torch.topk(logits, k, dim=-1)
                    logits = logits.masked_fill(
                        logits < top_values[:, [-1]],
                        float("-inf"),
                    )
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            prev_finished = finished.clone()
            if stop_id is not None:
                if prev_finished.any():
                    idx_next = torch.where(
                        prev_finished[:, None],
                        torch.full_like(idx_next, pad_token_id),
                        idx_next,
                    )
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            idx = torch.cat((idx, idx_next), dim=1)
            next_mask = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            if prev_finished.any():
                next_mask[prev_finished] = 0
            attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            if stop_id is not None and finished.all():
                break

        return idx[:, start_index:]


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 3], [4, 5, 6, 3], [7, 8, 9, 4]])
    config = ModelConfig(
        vocab_size=1024,
        max_seq_len=256,
        dim=512,
        n_heads=8,
        n_layers=6,
        n_kv_heads=4,
    )
    model = Transformer(config=config)
    out = model(x)

    print("input shape:", x.shape)
    print("output shape:", out["logits"].shape)
    print("model dim:", model.dim)
    print("num layers:", model.n_layers)
    print("vocab size:", model.vocab_size)
    print("max_seq_len:", model.max_seq_len)
    print("norm eps:", model.norm_eps)
