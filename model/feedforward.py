import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()

        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}")
        if d_ff < 1:
            raise ValueError(f"d_ff must be at least 1, got {d_ff}")
        if d_ff <= d_model:
            raise ValueError(f"d_ff={d_ff} must be greater than d_model={d_model}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        logger.info(f"SwiGLUFeedForward initialized — d_model={d_model}, d_ff={d_ff}, bias={bias}")

    def _init_weights(self):
        try:
            for layer in [self.w_gate, self.w_up, self.w_down]:
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        except Exception as e:
            logger.error(f"SwiGLUFeedForward weight initialization failed: {e}")
            raise

    def _swish(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return x * torch.sigmoid(x)

        except Exception as e:
            logger.error(f"Swish activation failed: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [batch, seq_len, d_model], got {x.shape}")

        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim={self.d_model}, got {x.shape[-1]}")

        try:
            gate = self._swish(self.w_gate(x))

        except Exception as e:
            logger.error(f"Gate projection failed: {e}")
            raise

        try:
            up = self.w_up(x)

        except Exception as e:
            logger.error(f"Up projection failed: {e}")
            raise

        try:
            fused = gate * up

        except Exception as e:
            logger.error(f"SwiGLU gate fusion failed: {e}")
            raise

        try:
            out = self.dropout(self.w_down(fused))

            if torch.isnan(out).any():
                logger.warning("NaN detected in FFN output — check for exploding activations")

            return out

        except Exception as e:
            logger.error(f"Down projection failed: {e}")
            raise