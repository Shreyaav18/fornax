from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime


class GenerateRequest(BaseModel):
    run_id: int
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    greedy: bool = False
    repetition_penalty: float = 1.0

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt must not be empty")
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def max_new_tokens_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("max_new_tokens must be at least 1")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_positive(cls, v):
        if v <= 0.0:
            raise ValueError("temperature must be positive")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("top_p")
    @classmethod
    def top_p_must_be_valid(cls, v):
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def repetition_penalty_must_be_positive(cls, v):
        if v <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        return v


class GenerateResponse(BaseModel):
    id: int
    run_id: int
    prompt: str
    output: str
    max_new_tokens: int
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    greedy: bool
    repetition_penalty: float
    generated_at: datetime


class RunResponse(BaseModel):
    id: int
    name: str
    status: str
    best_val_loss: Optional[float]
    final_step: Optional[int]
    started_at: datetime
    finished_at: Optional[datetime]
    checkpoint_path: Optional[str]
    notes: Optional[str]
    model_config: dict
    train_config: dict