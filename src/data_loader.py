# src/data_loader.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import pandas as pd

PAD_ID = 0  # reserved
# MASK_ID will be computed as num_items + 1 after building the vocab


def load_ratings(ratings_path: str) -> pd.DataFrame:
    """
    Load MovieLens ratings and sort interactions chronologically per user.

    Args:
        ratings_path: Path to MovieLens data with columns [userId, movieId, rating, timestamp].
    Returns:
        sorted_interactions: DataFrame with columns [userId, movieId, timestamp], sorted by (userId, timestamp).

    """
    
    sorted_interactions = pd.read_csv(ratings_path)
    # keep only needed columns; enforce types
    sorted_interactions = sorted_interactions[["userId", "movieId", "timestamp"]].copy()
    sorted_interactions["userId"] = sorted_interactions["userId"].astype(int)
    sorted_interactions["movieId"] = sorted_interactions["movieId"].astype(int)
    sorted_interactions["timestamp"] = sorted_interactions["timestamp"].astype(int)
    # chronological order
    sorted_interactions = sorted_interactions.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    return sorted_interactions


def build_item_vocab(
    sorted_interactions: pd.DataFrame
) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    """
    Build a dense item vocabulary from ALL interactions (covers train and test).

    Args:
        sorted_interactions: Interactions with at least column `movieId`.
    Returns:
        id_by_item: Mapping raw MovieLens movieId -> dense id in [1..num_items].
        item_by_id: Inverse mapping dense id -> raw movieId.
        num_items: Number of unique items in the corpus.
        mask_id: Special token id reserved for masking during training (= num_items + 1).
    """
    
    unique_items = sorted_interactions["movieId"].unique().tolist()
    unique_items.sort()
    id_by_item = {item: i + 1 for i, item in enumerate(unique_items)}  # 1, 2, ..., num_items
    item_by_id = {v: k for k, v in id_by_item.items()}
    num_items = len(unique_items)
    mask_id = num_items + 1
    return id_by_item, item_by_id, num_items, mask_id


def to_user_sequences(
    sorted_interactions: pd.DataFrame, 
    id_by_item: Dict[int, int]
) -> Dict[int, List[int]]:
    """
    Convert interactions to per-user sequences of dense item ids in chronological order.

    Args:
        sorted_interactions: Interactions sorted by (userId, timestamp).
        id_by_item: Mapping raw movieId -> dense id in [1, 2, ..., num_items].
    Returns:
        Dictionary mapping userId -> list of dense item ids (chronological).
    """
    
    sorted_interactions = sorted_interactions.copy()
    sorted_interactions["itemId"] = sorted_interactions["movieId"].map(id_by_item)
    # drop any interactions whose items weren't mapped (shouldn't happen)
    sorted_interactions = sorted_interactions.dropna(subset=["itemId"])
    sorted_interactions["itemId"] = sorted_interactions["itemId"].astype(int)
    seqs: Dict[int, List[int]] = (
        sorted_interactions.groupby("userId")["itemId"].apply(list).to_dict()
    )
    return seqs


def truncate_right(seq: List[int], max_len: int) -> List[int]:
    """
    Keep only the most recent `max_len` items from a sequence.

    Args:
        seq: Sequence of dense item ids.
        max_len: Maximum allowed length.
    Returns:
        Possibly shortened sequence (right-truncated).
    """
    
    if len(seq) <= max_len:
        return seq
    return seq[-max_len:]


def make_train_test_sequences(
    user_sequences: Dict[int, List[int]],
    max_len: int,
    min_user_interactions: int = 5,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Construct leave-one-out train/test sequences with right-truncation.

    Args:
        user_sequences: Mapping userId -> chronological dense item-id sequence.
        max_len: Max sequence length after truncation (keep most recent).
        min_user_interactions: Minimum interactions required to keep a user.
    Returns:
        train_sequences: Each sequence is the user's interactions excluding the last item.
        test_sequences: Each sequence is the user's full interactions.
    """
    
    train_sequences: List[List[int]] = []
    test_sequences: List[List[int]] = []

    for _, seq in user_sequences.items():
        if len(seq) < min_user_interactions:
            continue

        test_seq = truncate_right(seq, max_len)
        train_seq = truncate_right(seq[:-1], max_len)  # remove last

        if len(train_seq) == 0:
            continue

        train_sequences.append(train_seq)
        test_sequences.append(test_seq)

    return train_sequences, test_sequences


def prepare_movielens_sequences(
    ratings_path: str,
    max_len: int = 50,
    min_user_interactions: int = 5,
) -> Dict[str, Any]:
    """
    High-level preprocessing pipeline for BERT4Rec using MovieLens ratings.

    Args:
        ratings_path: Path to MovieLens data.
        max_len: Maximum sequence length after truncation (keep most recent).
        min_user_interactions: Drop users with fewer interactions than this threshold.
    Returns:
        a dictionary containing:
            - "train_sequences"
            - "test_sequences"
            - "num_items"
            - "mask_id"
            - "id_by_item"
            - "item_by_id"
            - "stats"
            - "pad_id"
    """
    
    sorted_interactions = load_ratings(ratings_path)
    id_by_item, item_by_id, num_items, mask_id = build_item_vocab(sorted_interactions)
    user_sequences = to_user_sequences(sorted_interactions, id_by_item)
    train_sequences, test_sequences = make_train_test_sequences(
        user_sequences, max_len=max_len, min_user_interactions=min_user_interactions
    )

    stats = {
        "users_total": len(user_sequences),
        "users_kept": len(train_sequences),
        "num_items": num_items,
        "avg_train_len": round(sum(map(len, train_sequences)) / max(1, len(train_sequences)), 2),
        "avg_test_len": round(sum(map(len, test_sequences)) / max(1, len(test_sequences)), 2),
        "max_len": max_len,
        "min_user_interactions": min_user_interactions,
    }

    return {
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "num_items": num_items,
        "mask_id": mask_id,
        "id_by_item": id_by_item,
        "item_by_id": item_by_id,
        "stats": stats,
        "pad_id": PAD_ID,
    }
