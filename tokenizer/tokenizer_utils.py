from tokenizers import Tokenizer
from typing import List
import os

def load_tokenizer(save_dir: str) -> Tokenizer:
    path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No tokenizer found at {path}")
    return Tokenizer.from_file(path)

def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text).ids

def encode_batch(tokenizer: Tokenizer, texts: List[str]) -> List[List[int]]:
    return [enc.ids for enc in tokenizer.encode_batch(texts)]

def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

def get_vocab_size(tokenizer: Tokenizer) -> int:
    return tokenizer.get_vocab_size()

def get_special_token_id(tokenizer: Tokenizer, token: str) -> int:
    return tokenizer.token_to_id(token)