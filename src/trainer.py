# src/trainer.py
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import nn
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    mlm_prob: float = 0.15,
) -> Dict[str, float]:
    """
    Train the BERT4Rec model for a single epoch.

    Args:
        model: The BERT4Rec model to train.
        loader: DataLoader providing batches of tokenized and masked input sequences
            and their corresponding target labels.
        optimizer: Optimizer instance for updating model parameters.
        device: Device on which to run computations ('cpu' or 'cuda').
        mlm_prob: probability of a given token being masked.

    Returns:
        The average training loss over the epoch.
    """
    
    model.train()
    total_loss, steps = 0.0, 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        masked_inputs, labels = model.mask_inputs(batch, mlm_prob=mlm_prob)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(masked_inputs, labels=labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return {"loss": total_loss / max(steps, 1)}


@torch.no_grad()
def evaluate_mlm_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mlm_prob: float = 0.15,
) -> Dict[str, float]:
    """
    Evaluate the BERT4Rec model on a dataset using MLM loss.

    Args:
        model: The trained (or partially trained) BERT4Rec model.
        loader: DataLoader providing batches of tokenized input sequences
            and their corresponding target labels.
        device: Device on which to run computations ('cpu' or 'cuda').
        mlm_prob: probability of a given token being masked.

    Returns:
        The average MLM loss across all evaluation batches.
    """

    model.eval()
    total_loss, steps = 0.0, 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        masked_inputs, labels = model.mask_inputs(batch, mlm_prob=mlm_prob)
        _, loss = model(masked_inputs, labels=labels)
        total_loss += float(loss.item())
        steps += 1

    return {"loss": total_loss / max(steps, 1)}
