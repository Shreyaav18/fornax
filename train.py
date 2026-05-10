import logging
import os
import asyncio
import torch
from dotenv import load_dotenv
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from tokenizer.train_tokenizer import train_bpe
from tokenizer.tokenizer_utils import load_tokenizer, get_vocab_size
from data.preprocess import load_corpus, tokenize_corpus, chunk_tokens, train_val_split
from data.dataset import TextDataset, build_dataloader
from model.transformer import GPT
from training.trainer import Trainer
from db.database import AsyncSessionFactory, init_db
from db.crud import create_training_run, update_checkpoint, complete_training_run, fail_training_run

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.txt")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "tokenizer_data")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
RESUME_CHECKPOINT = os.getenv("RESUME_CHECKPOINT", None)


async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_config = ModelConfig()
    train_config = TrainConfig(checkpoint_dir=CHECKPOINT_DIR)

    logger.info("Model config loaded")
    logger.info("Train config loaded")

    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        logger.info("No tokenizer found — training BPE tokenizer")
        try:
            tokenizer = train_bpe(
                corpus_path=CORPUS_PATH,
                vocab_size=model_config.vocab_size,
                save_dir=TOKENIZER_DIR
            )
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise
    else:
        logger.info("Tokenizer found — loading existing tokenizer")
        try:
            tokenizer = load_tokenizer(TOKENIZER_DIR)
        except Exception as e:
            logger.error(f"Tokenizer loading failed: {e}")
            raise

    actual_vocab_size = get_vocab_size(tokenizer)
    if actual_vocab_size != model_config.vocab_size:
        logger.warning(
            f"Tokenizer vocab size {actual_vocab_size} differs from config {model_config.vocab_size} — updating config"
        )
        model_config = model_config.model_copy(update={"vocab_size": actual_vocab_size})

    try:
        corpus = load_corpus(CORPUS_PATH)
        token_ids = tokenize_corpus(corpus, tokenizer)
        chunks = chunk_tokens(token_ids, model_config.max_seq_len)
        train_chunks, val_chunks = train_val_split(chunks)
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        raise

    try:
        train_dataset = TextDataset(train_chunks)
        val_dataset = TextDataset(val_chunks)

        train_loader = build_dataloader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda")
        )
    except Exception as e:
        logger.error(f"Dataset/DataLoader construction failed: {e}")
        raise

    try:
        model = GPT(model_config)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            model_config=model_config,
            device=device
        )
    except Exception as e:
        logger.error(f"Trainer initialization failed: {e}")
        raise

    if RESUME_CHECKPOINT:
        try:
            trainer.load_checkpoint(RESUME_CHECKPOINT)
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint {RESUME_CHECKPOINT}: {e}")
            raise

    await init_db()

    async with AsyncSessionFactory() as session:
        try:
            run = await create_training_run(
                session=session,
                model_config=model_config.model_dump(),
                train_config=train_config.model_dump()
            )
            run_id = run.id
        except Exception as e:
            logger.error(f"Failed to create training run record: {e}")
            raise

    original_save_checkpoint = trainer.save_checkpoint

    async def save_checkpoint_with_db(step: int, val_loss: float):
        original_save_checkpoint(step, val_loss)
        checkpoint_path = os.path.join(train_config.checkpoint_dir, f"fornax_step_{step}.pt")
        async with AsyncSessionFactory() as session:
            try:
                await update_checkpoint(
                    session=session,
                    run_id=run_id,
                    checkpoint_path=checkpoint_path,
                    best_val_loss=val_loss,
                    step=step
                )
            except Exception as e:
                logger.warning(f"DB checkpoint update failed at step {step}: {e}")

    trainer.save_checkpoint = lambda step, val_loss: asyncio.ensure_future(
        save_checkpoint_with_db(step, val_loss)
    )

    try:
        trainer.run()

        async with AsyncSessionFactory() as session:
            await complete_training_run(
                session=session,
                run_id=run_id,
                final_step=trainer.current_step,
                best_val_loss=trainer.best_val_loss
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user — saving emergency checkpoint")
        try:
            original_save_checkpoint(trainer.current_step, float("inf"))
        except Exception as e:
            logger.error(f"Emergency checkpoint save failed: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        async with AsyncSessionFactory() as session:
            try:
                await fail_training_run(session=session, run_id=run_id)
            except Exception as db_err:
                logger.error(f"Failed to mark run as failed in DB: {db_err}")
        raise


if __name__ == "__main__":
    asyncio.run(main())