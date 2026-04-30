import logging
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits shape [batch, seq_len, vocab_size], got {logits.shape}")

    if targets.dim() != 2:
        raise ValueError(f"Expected targets shape [batch, seq_len], got {targets.shape}")

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits batch/seq dims {logits.shape[:2]} do not match targets shape {targets.shape}"
        )

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

    try:
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        if torch.isnan(loss):
            logger.warning("NaN detected in loss — possible bad logits or all-ignored targets")

        if torch.isinf(loss):
            logger.warning("Inf detected in loss — possible extreme logit values")

        return loss

    except Exception as e:
        logger.error(f"Cross entropy loss computation failed: {e}")
        raise


def compute_perplexity(loss: torch.Tensor) -> float:
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("Cannot compute perplexity from NaN or Inf loss, returning inf")
        return float("inf")

    try:
        return torch.exp(loss).item()

    except Exception as e:
        logger.error(f"Perplexity computation failed: {e}")
        raise