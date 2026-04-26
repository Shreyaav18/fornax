import os
import logging
from tokenizers import Tokenizer
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_tokenizer(save_dir: str) -> Tokenizer:
    path = os.path.join(save_dir, "tokenizer.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No tokenizer found at: {path}")

    try:
        tokenizer = Tokenizer.from_file(path)
        logger.info(f"Tokenizer loaded from: {path}")
        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer from {path}: {e}")
        raise

def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    if not text or not text.strip():
        logger.warning("Received empty or whitespace-only text for encoding, returning empty list")
        return []

    try:
        return tokenizer.encode(text).ids

    except Exception as e:
        logger.error(f"Encoding failed for input text: {e}")
        raise

def encode_batch(tokenizer: Tokenizer, texts: List[str]) -> List[List[int]]:
    if not texts:
        logger.warning("Received empty list for batch encoding, returning empty list")
        return []

    filtered = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

    if len(filtered) < len(texts):
        logger.warning(f"{len(texts) - len(filtered)} empty strings skipped in batch encoding")

    if not filtered:
        return []

    try:
        indices, valid_texts = zip(*filtered)
        encoded = [enc.ids for enc in tokenizer.encode_batch(list(valid_texts))]

        result = [[] for _ in range(len(texts))]
        for idx, enc in zip(indices, encoded):
            result[idx] = enc

        return result

    except Exception as e:
        logger.error(f"Batch encoding failed: {e}")
        raise

def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    if not ids:
        logger.warning("Received empty id list for decoding, returning empty string")
        return ""

    try:
        return tokenizer.decode(ids, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Decoding failed for ids: {e}")
        raise

def get_vocab_size(tokenizer: Tokenizer) -> int:
    try:
        size = tokenizer.get_vocab_size()
        logger.info(f"Vocab size: {size}")
        return size

    except Exception as e:
        logger.error(f"Failed to retrieve vocab size: {e}")
        raise

def get_special_token_id(tokenizer: Tokenizer, token: str) -> Optional[int]:
    try:
        token_id = tokenizer.token_to_id(token)

        if token_id is None:
            logger.warning(f"Special token '{token}' not found in vocabulary")

        return token_id

    except Exception as e:
        logger.error(f"Failed to get id for token '{token}': {e}")
        raise