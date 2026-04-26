import os
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_bpe(corpus_path: str, vocab_size: int, save_dir: str) -> Tokenizer:
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

    logger.info(f"Starting BPE training on corpus: {corpus_path}")
    logger.info(f"Target vocab size: {vocab_size}")

    try:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_frequency=2,
            show_progress=True
        )

        tokenizer.train(files=[corpus_path], trainer=trainer)
        logger.info("BPE training complete")

    except Exception as e:
        logger.error(f"BPE training failed: {e}")
        raise

    try:
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        if bos_id is None or eos_id is None:
            raise ValueError("Special tokens <bos> or <eos> missing from trained vocabulary")

        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", bos_id),
                ("<eos>", eos_id),
            ]
        )
        logger.info("Post processor configured successfully")

    except Exception as e:
        logger.error(f"Post processor setup failed: {e}")
        raise

    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "tokenizer.json")
        tokenizer.save(save_path)
        logger.info(f"Tokenizer saved to: {save_path}")

    except OSError as e:
        logger.error(f"Failed to save tokenizer to {save_dir}: {e}")
        raise

    return tokenizer