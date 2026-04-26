import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()

        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

        logger.info(f"RMSNorm initialized — d_model={d_model}, eps={eps}")

    def _compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        except Exception as e:
            logger.error(f"RMS computation failed: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"Expected input of at least 2 dims, got {x.dim()}")

        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim={self.d_model}, got {x.shape[-1]}")

        try:
            rms = self._compute_rms(x)
            x_normed = x / rms

        except Exception as e:
            logger.error(f"RMSNorm normalization failed: {e}")
            raise

        try:
            out = self.scale * x_normed

            if torch.isnan(out).any():
                logger.warning("NaN detected in RMSNorm output — possible zero RMS input")

            return out

        except Exception as e:
            logger.error(f"RMSNorm scaling failed: {e}")
            raise