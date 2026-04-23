from pydantic import BaseModel, model_validator

class ModelConfig(BaseModel):
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = False

    @model_validator(mode="after")
    def validate_dims(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_ff > self.d_model, "d_ff must be greater than d_model"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        return self