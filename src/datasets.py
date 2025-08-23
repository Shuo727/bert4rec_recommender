# src/datasets.py
from typing import List
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0  # must match model


class SequenceDataset(Dataset):
    """
    Takes a list of item-id sequences (each a List[int] with ids in [1..num_items]).
    Truncates to max_len from the right; no masking here (done in model).
    """
    
    def __init__(self, sequences: List[List[int]], max_len: int):
        self.max_len = max_len
        # keep non-empty sequences
        self.sequences = [seq for seq in sequences if len(seq) > 0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> torch.Tensor:
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]  # keep most recent interactions
        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Right-pad with PAD_ID=0 to max length in batch.
    """
    
    return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
