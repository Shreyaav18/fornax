import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from model.embeddings import RotaryEmbedding, apply_rotary_emb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

        self._init_weights()
        logger.info(f"MultiHeadSelfAttention initialized — d_model={d_model}, n_heads={n_heads}, head_dim={self.head_dim}")

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        try:
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            weights = F.softmax(scores, dim=-1)

            if torch.isnan(weights).any():
                logger.warning("NaN detected in attention weights — possible all-masked row, clamping to zero")
                weights = torch.nan_to_num(weights, nan=0.0)

            weights = self.attn_dropout(weights)
            return torch.matmul(weights, v)

        except Exception as e:
            logger.error(f"Scaled dot product attention failed: {e}")
            raise

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        try:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            return mask
        except Exception as e:
            logger.error(f"Causal mask construction failed: {e}")
            raise

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [batch, seq_len, d_model], got {x.shape}")

        batch_size, seq_len, _ = x.shape

        try:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        except Exception as e:
            logger.error(f"QKV projection failed: {e}")
            raise

        try:
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        except Exception as e:
            logger.error(f"QKV reshape failed: {e}")
            raise

        try:
            q, k = apply_rotary_emb(q, k, self.rope)

        except Exception as e:
            logger.error(f"RoPE application failed: {e}")
            raise

        if kv_cache is not None:
            try:
                if "k" in kv_cache and "v" in kv_cache:
                    k = torch.cat([kv_cache["k"], k], dim=2)
                    v = torch.cat([kv_cache["v"], v], dim=2)
                kv_cache = {"k": k, "v": v}

            except Exception as e:
                logger.error(f"KV cache update failed: {e}")
                raise

        if mask is None:
            full_seq_len = k.shape[2]
            mask = self._build_causal_mask(full_seq_len, device=x.device)
            if kv_cache is not None:
                mask = mask[:, :, -seq_len:, :]

        try:
            attn_output = self._scaled_dot_product(q, k, v, mask)

        except Exception as e:
            logger.error(f"Attention computation failed: {e}")
            raise

        try:
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.resid_dropout(self.out_proj(attn_output))
            return output, kv_cache

        except Exception as e:
            logger.error(f"Attention output projection failed: {e}")
            raise