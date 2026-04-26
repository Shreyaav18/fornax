import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, chunks: List[List[int]]):
        if not chunks:
            raise ValueError("Cannot create dataset from empty chunk list")

        try:
            self.data = [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]
            logger.info(f"TextDataset created with {len(self.data):,} samples")

        except Exception as e:
            logger.error(f"Failed to create TextDataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        try:
            chunk = self.data[idx]
            input_ids = chunk[:-1]
            target_ids = chunk[1:]
            return input_ids, target_ids

        except Exception as e:
            logger.error(f"Failed to retrieve item at index {idx}: {e}")
            raise

def build_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if len(dataset) < batch_size:
        logger.warning(f"Dataset size ({len(dataset)}) is smaller than batch_size ({batch_size})")

    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        logger.info(f"DataLoader built — {len(loader):,} batches, batch_size={batch_size}, shuffle={shuffle}")
        return loader

    except Exception as e:
        logger.error(f"Failed to build DataLoader: {e}")
        raise