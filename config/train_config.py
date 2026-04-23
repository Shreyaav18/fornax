from pydantic import BaseModel, model_validator

class TrainConfig(BaseModel):
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "transformer-from-scratch"
    eval_interval: int = 500
    checkpoint_interval: int = 500
    gradient_accumulation_steps: int = 1

    @model_validator(mode="after")
    def validate_lr(self):
        assert self.min_lr < self.lr, "min_lr must be less than lr"
        assert self.warmup_steps < self.max_steps, "warmup_steps must be less than max_steps"
        return self