import logging
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from model.transformer import GPT
from tokenizer.tokenizer_utils import decode, encode, get_special_token_id
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    try:
        return logits / temperature
    except Exception as e:
        logger.error(f"Temperature scaling failed: {e}")
        raise


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k < 1:
        raise ValueError(f"top_k must be at least 1, got {k}")

    if k >= logits.shape[-1]:
        logger.warning(f"top_k={k} >= vocab_size={logits.shape[-1]}, top_k filtering has no effect")
        return logits

    try:
        values, _ = torch.topk(logits, k, dim=-1)
        min_val = values[..., -1].unsqueeze(-1)
        return logits.masked_fill(logits < min_val, float("-inf"))

    except Exception as e:
        logger.error(f"Top-k filtering failed: {e}")
        raise


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")

    if p == 1.0:
        return logits

    try:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
        return logits

    except Exception as e:
        logger.error(f"Top-p filtering failed: {e}")
        raise


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argmax(logits, dim=-1)
    except Exception as e:
        logger.error(f"Greedy decode failed: {e}")
        raise


def sample_token(logits: torch.Tensor) -> torch.Tensor:
    try:
        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logger.warning("Invalid probabilities detected before sampling — falling back to uniform distribution")
            probs = torch.ones_like(probs) / probs.shape[-1]

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    except Exception as e:
        logger.error(f"Token sampling failed: {e}")
        raise


def repetition_penalty_apply(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float
) -> torch.Tensor:
    if penalty == 1.0:
        return logits

    if penalty <= 0.0:
        raise ValueError(f"repetition_penalty must be positive, got {penalty}")

    try:
        for token_id in set(generated_ids):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits

    except Exception as e:
        logger.error(f"Repetition penalty application failed: {e}")
        raise


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

    if top_k is not None and top_p is not None:
        logger.info(f"Both top_k={top_k} and top_p={top_p} set — applying top_k first, then top_p")

    model.eval()

    try:
        eos_id = get_special_token_id(tokenizer, "<eos>")
        input_ids = encode(tokenizer, prompt)

        if not input_ids:
            raise ValueError("Prompt encoded to empty token list")

        logger.info(f"Generating — prompt tokens: {len(input_ids)}, max_new_tokens: {max_new_tokens}")

    except Exception as e:
        logger.error(f"Prompt encoding failed: {e}")
        raise

    try:
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    except Exception as e:
        logger.error(f"Failed to create input tensor: {e}")
        raise

    kv_cache = model.reset_kv_cache()
    generated_ids = list(input_ids)

    try:
        logits, kv_cache = model(input_tensor, kv_cache=kv_cache)
        next_logits = logits[:, -1, :]

    except Exception as e:
        logger.error(f"Initial forward pass failed: {e}")
        raise

    for step in range(max_new_tokens):
        try:
            if greedy:
                next_token_id = greedy_decode(next_logits).item()
            else:
                scaled = temperature_scale(next_logits, temperature)

                if top_k is not None:
                    scaled = top_k_filter(scaled, top_k)

                if top_p is not None:
                    scaled = top_p_filter(scaled, top_p)

                if repetition_penalty != 1.0:
                    scaled = repetition_penalty_apply(scaled, generated_ids, repetition_penalty)

                next_token_id = sample_token(scaled).item()

        except Exception as e:
            logger.error(f"Token selection failed at step {step}: {e}")
            raise

        generated_ids.append(next_token_id)

        if eos_id is not None and next_token_id == eos_id:
            logger.info(f"EOS token generated at step {step} — stopping early")
            break

        try:
            next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            logits, kv_cache = model(next_input, kv_cache=kv_cache)
            next_logits = logits[:, -1, :]

        except Exception as e:
            logger.error(f"Forward pass failed at generation step {step}: {e}")
            raise

    try:
        new_ids = generated_ids[len(input_ids):]
        output_text = decode(tokenizer, new_ids)
        logger.info(f"Generation complete — {len(new_ids)} new tokens generated")
        return output_text

    except Exception as e:
        logger.error(f"Decoding generated tokens failed: {e}")
        raise