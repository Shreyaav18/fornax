import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from config.train_config import TrainConfig
from config.model_config import ModelConfig
from model.transformer import GPT
from training.losses import cross_entropy_loss, compute_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available — training metrics will only be logged locally")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    max_lr: float
):
    if warmup_steps >= max_steps:
        raise ValueError(f"warmup_steps={warmup_steps} must be less than max_steps={max_steps}")

    try:
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + cosine_decay * (max_lr - min_lr)) / max_lr

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    except Exception as e:
        logger.error(f"LR scheduler creation failed: {e}")
        raise


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config: TrainConfig,
        model_config: ModelConfig,
        device: torch.device
    ):
        if not isinstance(model, GPT):
            raise TypeError(f"Expected GPT model instance, got {type(model)}")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_config = train_config
        self.model_config = model_config
        self.device = device
        self.current_step = 0
        self.best_val_loss = float("inf")

        try:
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to device: {device}")
        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {e}")
            raise

        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.info("AdamW optimizer initialized")
        except Exception as e:
            logger.error(f"Optimizer initialization failed: {e}")
            raise

        try:
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                warmup_steps=train_config.warmup_steps,
                max_steps=train_config.max_steps,
                min_lr=train_config.min_lr,
                max_lr=train_config.lr
            )
            logger.info("LR scheduler initialized")
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {e}")
            raise

        if WANDB_AVAILABLE and train_config.wandb_project:
            try:
                wandb.init(
                    project=train_config.wandb_project,
                    name="fornax",
                    config={
                        **train_config.model_dump(),
                        **model_config.model_dump()
                    }
                )
                logger.info(f"wandb initialized — project: {train_config.wandb_project}")
            except Exception as e:
                logger.warning(f"wandb initialization failed, continuing without it: {e}")

        os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    def _to_device(self, batch):
        try:
            input_ids, targets = batch
            return input_ids.to(self.device), targets.to(self.device)
        except Exception as e:
            logger.error(f"Failed to move batch to device: {e}")
            raise

    def train_step(self, batch) -> float:
        self.model.train()

        try:
            input_ids, targets = self._to_device(batch)
        except Exception as e:
            logger.error(f"Batch device transfer failed at step {self.current_step}: {e}")
            raise

        try:
            logits, _ = self.model(input_ids)
            loss = cross_entropy_loss(logits, targets, label_smoothing=0.1)
            loss = loss / self.train_config.gradient_accumulation_steps

        except Exception as e:
            logger.error(f"Forward pass failed at step {self.current_step}: {e}")
            raise

        try:
            loss.backward()
        except Exception as e:
            logger.error(f"Backward pass failed at step {self.current_step}: {e}")
            raise

        return loss.item() * self.train_config.gradient_accumulation_steps

    @torch.no_grad()
    def eval_step(self, batch) -> float:
        self.model.eval()

        try:
            input_ids, targets = self._to_device(batch)
            logits, _ = self.model(input_ids)
            loss = cross_entropy_loss(logits, targets)
            return loss.item()

        except Exception as e:
            logger.error(f"Eval step failed: {e}")
            raise

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        logger.info("Running evaluation...")
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        try:
            for batch in self.val_loader:
                loss = self.eval_step(batch)
                total_loss += loss
                total_batches += 1

            if total_batches == 0:
                raise ValueError("Validation loader produced no batches")

            avg_loss = total_loss / total_batches
            perplexity = compute_perplexity(torch.tensor(avg_loss))

            logger.info(f"Eval — loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
            return {"val_loss": avg_loss, "val_perplexity": perplexity}

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_checkpoint(self, step: int, val_loss: float):
        try:
            checkpoint = {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "model_config": self.model_config.model_dump(),
                "train_config": self.train_config.model_dump()
            }

            path = os.path.join(self.train_config.checkpoint_dir, f"fornax_step_{step}.pt")
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved at step {step} — path: {path}, val_loss: {val_loss:.4f}")

        except Exception as e:
            logger.error(f"Checkpoint save failed at step {step}: {e}")
            raise

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.current_step = checkpoint["step"]
            self.best_val_loss = checkpoint.get("val_loss", float("inf"))
            logger.info(f"Checkpoint loaded from {path} — resuming from step {self.current_step}")

        except Exception as e:
            logger.error(f"Checkpoint load failed from {path}: {e}")
            raise

    def run(self):
        logger.info(f"Starting training — max_steps={self.train_config.max_steps}, device={self.device}")

        accumulation_loss = 0.0
        self.optimizer.zero_grad()

        train_iter = iter(self.train_loader)

        while self.current_step < self.train_config.max_steps:
            for accum_step in range(self.train_config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                try:
                    loss = self.train_step(batch)
                    accumulation_loss += loss
                except Exception as e:
                    logger.error(f"Train step failed at step {self.current_step}: {e}")
                    raise

            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )
            except Exception as e:
                logger.error(f"Gradient clipping failed at step {self.current_step}: {e}")
                raise

            try:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            except Exception as e:
                logger.error(f"Optimizer step failed at step {self.current_step}: {e}")
                raise

            current_lr = self.scheduler.get_last_lr()[0]
            avg_loss = accumulation_loss / self.train_config.gradient_accumulation_steps

            if self.current_step % 50 == 0:
                logger.info(
                    f"Step {self.current_step}/{self.train_config.max_steps} — "
                    f"loss: {avg_loss:.4f}, lr: {current_lr:.6f}, grad_norm: {grad_norm:.4f}"
                )

            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm,
                        "step": self.current_step
                    })
                except Exception as e:
                    logger.warning(f"wandb logging failed at step {self.current_step}: {e}")

            accumulation_loss = 0.0

            if self.current_step % self.train_config.eval_interval == 0 and self.current_step > 0:
                try:
                    eval_metrics = self.evaluate()

                    if WANDB_AVAILABLE and wandb.run is not None:
                        try:
                            wandb.log({**eval_metrics, "step": self.current_step})
                        except Exception as e:
                            logger.warning(f"wandb eval logging failed: {e}")

                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]
                        self.save_checkpoint(self.current_step, self.best_val_loss)
                        logger.info(f"New best val_loss: {self.best_val_loss:.4f} — checkpoint saved")

                except Exception as e:
                    logger.error(f"Evaluation/checkpoint failed at step {self.current_step}: {e}")
                    raise

            if self.current_step % self.train_config.checkpoint_interval == 0 and self.current_step > 0:
                try:
                    self.save_checkpoint(self.current_step, avg_loss)
                except Exception as e:
                    logger.warning(f"Periodic checkpoint failed at step {self.current_step}: {e}")

            self.current_step += 1

        logger.info(f"Training complete — best val_loss: {self.best_val_loss:.4f}")

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()