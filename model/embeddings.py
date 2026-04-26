import logging
import torch
import torch.nn as nn
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        if vocab_size < 1:
            raise ValueError(f"vocab_size must be at least 1, got {vocab_size}")
        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        logger.info(f"TokenEmbedding initialized — vocab_size={vocab_size}, d_model={d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            raise TypeError(f"Expected input dtype torch.long, got {x.dtype}")

        if x.max() >= self.embedding.num_embeddings:
            raise ValueError(f"Token id {x.max().item()} exceeds vocab_size {self.embedding.num_embeddings}")

        try:
            return self.embedding(x)
        except Exception as e:
            logger.error(f"TokenEmbedding forward failed: {e}")
            raise


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dim must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        try:
            theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("theta", theta)

            self._build_cache(max_seq_len)
            logger.info(f"RotaryEmbedding initialized — dim={dim}, max_seq_len={max_seq_len}, base={base}")

        except Exception as e:
            logger.error(f"RotaryEmbedding initialization failed: {e}")
            raise

    def _build_cache(self, seq_len: int):
        try:
            positions = torch.arange(seq_len, device=self.theta.device).float()
            freqs = torch.outer(positions, self.theta)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos_cache", emb.cos())
            self.register_buffer("sin_cache", emb.sin())

        except Exception as e:
            logger.error(f"RoPE cache build failed: {e}")
            raise

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            logger.warning(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}, rebuilding RoPE cache")
            try:
                self._build_cache(seq_len)
            except Exception as e:
                logger.error(f"RoPE cache rebuild failed for seq_len={seq_len}: {e}")
                raise

        try:
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            return (x * cos) + (self._rotate_half(x) * sin)

        except Exception as e:
            logger.error(f"RotaryEmbedding forward failed: {e}")
            raise


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryEmbedding
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        q_rot = rope(q)
        k_rot = rope(k)
        return q_rot, k_rot

    except Exception as e:
        logger.error(f"apply_rotary_emb failed: {e}")
        raise