import os
import logging
from typing import List, Tuple
from tokenizers import Tokenizer
from tokenizer.tokenizer_utils import encode, load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_corpus(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            raise ValueError(f"Corpus file at {path} is empty")

        logger.info(f"Loaded corpus from {path} — {len(text):,} characters")
        return text

    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading corpus at {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        raise

def tokenize_corpus(text: str, tokenizer: Tokenizer) -> List[int]:
    if not text or not text.strip():
        raise ValueError("Cannot tokenize empty text")

    try:
        logger.info("Tokenizing corpus...")
        token_ids = encode(tokenizer, text)

        if not token_ids:
            raise ValueError("Tokenization produced empty token list")

        logger.info(f"Tokenization complete — {len(token_ids):,} tokens")
        return token_ids

    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise

def chunk_tokens(token_ids: List[int], seq_len: int) -> List[List[int]]:
    if not token_ids:
        raise ValueError("Cannot chunk empty token list")

    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")

    try:
        chunks = [
            token_ids[i: i + seq_len + 1]
            for i in range(0, len(token_ids) - seq_len, seq_len)
        ]

        chunks = [c for c in chunks if len(c) == seq_len + 1]

        if not chunks:
            raise ValueError(f"No complete chunks produced — corpus may be too small for seq_len={seq_len}")

        logger.info(f"Produced {len(chunks):,} chunks of length {seq_len + 1}")
        return chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise

def train_val_split(chunks: List[List[int]], val_ratio: float = 0.1) -> Tuple[List, List]:
    if not chunks:
        raise ValueError("Cannot split empty chunk list")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    try:
        split_idx = int(len(chunks) * (1 - val_ratio))

        if split_idx == 0 or split_idx == len(chunks):
            raise ValueError(f"val_ratio={val_ratio} produces an empty train or val split")

        train_chunks = chunks[:split_idx]
        val_chunks = chunks[split_idx:]

        logger.info(f"Split — train: {len(train_chunks):,} chunks, val: {len(val_chunks):,} chunks")
        return train_chunks, val_chunks

    except Exception as e:
        logger.error(f"Train/val split failed: {e}")
        raise