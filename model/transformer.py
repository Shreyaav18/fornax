import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from config.model_config import ModelConfig
from model.embeddings import TokenEmbedding
from model.block import TransformerBlock
from model.normalization import RMSNorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        try:
            self.token_embedding = TokenEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model
            )
        except Exception as e:
            logger.error(f"Failed to initialize token embedding: {e}")
            raise

        try:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    bias=config.bias,
                    layer_idx=i
                )
                for i in range(config.n_layers)
            ])
        except Exception as e:
            logger.error(f"Failed to initialize transformer blocks: {e}")
            raise

        try:
            self.final_norm = RMSNorm(config.d_model)
        except Exception as e:
            logger.error(f"Failed to initialize final RMSNorm: {e}")
            raise

        try:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        except Exception as e:
            logger.error(f"Failed to initialize output projection: {e}")
            raise

        if config.tie_weights:
            self.output_projection.weight = self.token_embedding.embedding.weight
            logger.info("Weight tying enabled — output projection shares weights with token embedding")

        self._init_weights()
        total_params = self.get_num_params()
        trainable_params = self.get_num_params(trainable_only=True)
        logger.info(f"GPT initialized — total params: {total_params:,}, trainable: {trainable_params:,}")
        logger.info(f"Config — layers={config.n_layers}, heads={config.n_heads}, d_model={config.d_model}, d_ff={config.d_ff}")

    def _init_weights(self):
        try:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.scale)
            logger.info("GPT weights initialized")

        except Exception as e:
            logger.error(f"Weight initialization failed: {e}")
            raise

    def get_num_params(self, trainable_only: bool = False) -> int:
        try:
            if trainable_only:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            return sum(p.numel() for p in self.parameters())

        except Exception as e:
            logger.error(f"Parameter count failed: {e}")
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq_len], got {input_ids.shape}")

        _, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        if kv_cache is not None and len(kv_cache) != self.config.n_layers:
            raise ValueError(
                f"kv_cache length {len(kv_cache)} does not match n_layers {self.config.n_layers}"
            )

        try:
            x = self.token_embedding(input_ids)
        except Exception as e:
            logger.error(f"Token embedding failed: {e}")
            raise

        updated_kv_cache = []

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None

            try:
                x, layer_cache = block(x, mask=mask, kv_cache=layer_cache)
                updated_kv_cache.append(layer_cache)

            except Exception as e:
                logger.error(f"Forward pass failed at block {i}: {e}")
                raise

        try:
            x = self.final_norm(x)
        except Exception as e:
            logger.error(f"Final RMSNorm failed: {e}")
            raise

        try:
            logits = self.output_projection(x)
        except Exception as e:
            logger.error(f"Output projection failed: {e}")
            raise

        if torch.isnan(logits).any():
            logger.warning("NaN detected in output logits — check for training instability or bad inputs")

        return_cache = updated_kv_cache if kv_cache is not None else None
        return logits, return_cache

    def reset_kv_cache(self) -> List[Dict]:
        return [{} for _ in range(self.config.n_layers)]