# src/eval.py
from __future__ import annotations
from typing import Dict, List, Iterable, Tuple
import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader


def _mask_last_position(
    batch: Tensor,
    mask_token_id: int,
    pad_id: int = 0
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Replace the last non-padding token in each sequence with a `[MASK]` token.
    This is used during evaluation to prepare inputs for next-item prediction.

    Args:
        batch: Input data of shape (batch_size, sequence_length).
        mask_token_id: Token ID used to represent `[MASK]`.
        pad_id: Token ID used to represent padding.
    Returns:
        input_masked: Tensor with the same shape as batch, but where the last
            non-padding item in each row is replaced with `mask_token_id`.
        labels: Tensor of shape (batch_size,) containing the original last item IDs.
        last_idx: Tensor of shape (batch_size,) with indices of last non-padding positions.
    """

    batch_size, seq_len = batch.shape
    
    is_nonpad = batch.ne(pad_id)
    # positions 0, ..., seq_len-1 -> give each weight for argmax
    idx = torch.arange(seq_len, device=batch.device).unsqueeze(0).expand(batch_size, seq_len)
    # set pad positions to -1 so they don't win argmax
    masked_idx_vals = idx * is_nonpad + (~is_nonpad) * (-1)
    # last index of non-pad per row
    last_idx = masked_idx_vals.argmax(dim=1)  # (batch_size,)

    input_masked = batch.clone()
    input_masked[torch.arange(batch_size, device=batch.device), last_idx] = mask_token_id

    labels = [batch[i, last_idx[i]].item() for i in range(batch_size)]
    labels = torch.tensor(labels, device=batch.device, dtype=torch.long)

    return input_masked, labels, last_idx


def _compute_metrics(
    items_pred: List[List[int]],
    ground_truth: List[int],
    ks: Tuple[int]=(5, 10)
) -> Dict[str, float]:
    """
    Compute Hit@K, Precision@K, NDCG@K, and MRR@K for ranked predictions.

    Args:
        items_pred: List of elements, each element is a ranked list of item ids.
        ground_truth: List, with the single relevant item id per example.
        ks: Cutoffs to evaluate.
    Returns:
        Global average metrics across N examples:
        "Hit@K", "Precision@K", "NDCG@K", "MRR@K" for each K in ks.
        - 'Hit@K': Fraction of cases where the ground-truth item 
            appears in the top-K ranked predictions.
        - 'Precision@K': Proportion of relevant items captured in the top-K over K.
        - 'NDCG@K': Normalized Discounted Cumulative Gain, which accounts 
            for the ranking position of the ground-truth item.
        - 'MRR@K': reciprocal rank = 1/rank of the ground-truth next item.
    """
    
    gt = np.asarray(list(ground_truth), dtype=np.int64)
    N = len(items_pred)
    assert N == len(gt), "items_pred and ground_truth must have the same length"

    # Precompute rank dicts (1-based); missing → inf
    rankers = [{item: r for r, item in enumerate(preds, start=1)} for preds in items_pred]

    # Mask invalid GT (<=0); we'll skip them in metrics
    valid = gt > 0
    if not np.any(valid):
        # Avoid divide-by-zero: return zeros
        return {m: 0.0 for K in ks for m in (f"Hit@{K}", f"Precision@{K}", f"NDCG@{K}", f"MRR@{K}")}

    metrics: Dict[str, float] = {}
    num_valid_target = int(valid.sum())

    for K in ks:
        hit = 0.0
        prec_sum = 0.0
        ndcg_sum = 0.0
        mrr_sum = 0.0

        for i in range(N):
            if not valid[i]:
                continue
            r = rankers[i].get(int(gt[i]), np.inf)  # 1-based rank
            if r <= K:
                hit += 1.0
                prec_sum += 1.0 / K                # single relevant → Precision@K = Hit@K / K
                ndcg_sum += 1.0 / np.log2(r + 1.0) # DCG with rel=1 at rank r
                mrr_sum += 1.0 / r                 # reciprocal rank

        metrics[f"Hit@{K}"] = hit / num_valid_target
        metrics[f"Precision@{K}"] = prec_sum / num_valid_target
        metrics[f"NDCG@{K}"] = ndcg_sum / num_valid_target
        metrics[f"MRR@{K}"] = mrr_sum / num_valid_target

    return metrics


@torch.no_grad()
def evaluate_model_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_token_id: int,
    pad_id: int = 0,
    items_only_head: bool = True,
    ks: Tuple[int]=(5, 10, 20),
) -> Dict[str, float]:
    """
    Evaluate the model using top-K recommendation metrics.

    Args:
        model: The trained recommendation model (e.g., BERT4Rec).
        loader: DataLoader yielding evaluation batches of (input_seq, target_item).
        device: Device on which to run inference.
        mask_token_id: mask_token_id.
        pad_id: pad_id.
        items_only_head:
            If True, head classes are 0..num_items-1.
            If False, classes already align to item ids (e.g., 0=PAD, 1, ..., num_items).
        ks: The numbers of top-ranked items to consider for evaluation.
    Returns:
        Dictionary containing top-K evaluation metrics: 
        "Hit@K", "Precision@K", "NDCG@K", "MRR@K"
    """

    model.eval()

    # Collect ground-truth labels (the last real item per sequence)
    all_gt: List[int] = []
    all_pred: List[List[int]] = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        
        inp, labels, last_idx = _mask_last_position(batch, mask_token_id, pad_id)
        
        logits, _ = model(inp, labels=None)
        # We only care about the masked position logits
        batch_size, seq_len, C = logits.shape
        row_idx = torch.arange(batch_size, device=device)
        pos_logits = logits[row_idx, last_idx, :]  # (batch_size, num_items+1)

        # Top-K per row
        topk_vals, topk_idx = torch.topk(pos_logits, k=max(ks), dim=-1)

        if items_only_head:
            topk_idx = topk_idx + 1  # map class→item id
 
        all_gt.extend(labels.tolist())
        all_pred.extend(topk_idx.tolist())

    return _compute_metrics(all_pred, all_gt, ks=ks)


def evaluate_baseline_topk(
    items_pred: List[List[int]],
    loader: DataLoader,
    device: torch.device,
    mask_token_id: int,
    pad_id: int = 0,
    ks: Tuple[int]=(5, 10, 20),
) -> Dict[str, float]:
    """
    Evaluate the items_pred using top-K recommendation metrics.

    Args:
        items_pred: precomputed ranked lists (baseline).
        loader: DataLoader yielding evaluation batches of (input_seq, target_item).
        device: Device on which to run inference.
        mask_token_id: mask_token_id.
        pad_id: pad_id.
        ks: The numbers of top-ranked items to consider for evaluation.
    Returns:
        Dictionary containing top-K evaluation metrics: 
        "Hit@K", "Precision@K", "NDCG@K", "MRR@K"
    """

    # Collect ground-truth labels (the last real item per sequence)
    all_gt: List[int] = []
    all_pred: List[List[int]] = []

    pred_iter = iter(items_pred)
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device) if isinstance(batch, Tensor) else torch.as_tensor(batch, device=device)
        
        _, labels, _ = _mask_last_position(batch, mask_token_id, pad_id)
        
        all_gt.extend(labels.tolist())

        # Pull predictions for each row in this batch
        batch_size = batch.size(0)
        for _ in range(batch_size):
            all_pred.append(list(next(pred_iter)))
    
    # Safety: ensure counts match
    assert len(all_pred) == len(all_gt), "items_pred length does not match number of examples in loader"
    return _compute_metrics(all_pred, all_gt, ks=ks)
