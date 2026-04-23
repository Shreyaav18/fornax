from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
import os

def train_bpe(corpus_path: str, vocab_size: int, save_dir: str):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ]
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    return tokenizer