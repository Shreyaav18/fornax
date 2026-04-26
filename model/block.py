import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from model.attention import MultiHeadSelfAttention
from model.feedforward import SwiGLUFeedForward
from model.normalization import RMSNorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False,
        layer_idx: int = 0
    ):
        super().__init__()

        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be at least 1, got {n_heads}")
        if d_ff < d_model:
            raise ValueError(f"d_ff={d_ff} must be greater than d_model={d_model}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.layer_idx = layer_idx

        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)

        try:
            self.attention = MultiHeadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                bias=bias
            )
        except Exception as e:
            logger.error(f"Failed to initialize attention in block {layer_idx}: {e}")
            raise

        try:
            self.ffn = SwiGLUFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                bias=bias
            )
        except Exception as e:
            logger.error(f"Failed to initialize FFN in block {layer_idx}: {e}")
            raise

        logger.info(f"TransformerBlock {layer_idx} initialized — d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        if x.dim() != 3:
            raise ValueError(f"Block {self.layer_idx}: expected input shape [batch, seq_len, d_model], got {x.shape}")

        if x.shape[-1] != self.norm_1.d_model:
            raise ValueError(
                f"Block {self.layer_idx}: input last dim {x.shape[-1]} "
                f"does not match d_model {self.norm_1.d_model}"
            )

        try:
            normed = self.norm_1(x)
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: pre-attention RMSNorm failed: {e}")
            raise

        try:
            attn_out, kv_cache = self.attention(normed, mask=mask, kv_cache=kv_cache)
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: attention forward failed: {e}")
            raise

        try:
            x = x + attn_out
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: attention residual addition failed: {e}")
            raise

        try:
            normed = self.norm_2(x)
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: pre-FFN RMSNorm failed: {e}")
            raise

        try:
            ffn_out = self.ffn(normed)
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: FFN forward failed: {e}")
            raise

        try:
            x = x + ffn_out
        except Exception as e:
            logger.error(f"Block {self.layer_idx}: FFN residual addition failed: {e}")
            raise

        if torch.isnan(x).any():
            logger.warning(f"Block {self.layer_idx}: NaN detected in block output — possible training instability")

        return x, kv_cache