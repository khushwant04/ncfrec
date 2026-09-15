"""Deterministic training and full-catalog ranking evaluation utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ncfrec.model import NeuMF


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: NeuMF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean binary cross-entropy loss."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0

    for users, items, labels in loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(users, items)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = len(labels)
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("training loader is empty")
    return total_loss / total_examples


@torch.inference_mode()
def evaluate_ranking(
    model: NeuMF,
    held_out: pd.DataFrame,
    blocked_items: Mapping[int, set[int]],
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute full-catalog Hit Rate and NDCG for one held-out item per user."""
    if k < 1:
        raise ValueError("k must be positive")
    if held_out.empty:
        raise ValueError("held_out interactions are empty")

    model.eval()
    all_items = torch.arange(model.num_items, dtype=torch.long, device=device)
    hits = 0.0
    ndcg = 0.0

    for row in held_out.itertuples(index=False):
        user_idx = int(row.user_idx)
        true_item = int(row.item_idx)
        blocked = set(blocked_items.get(user_idx, set()))
        blocked.discard(true_item)
        mask = torch.ones(model.num_items, dtype=torch.bool, device=device)
        if blocked:
            mask[torch.tensor(sorted(blocked), dtype=torch.long, device=device)] = False
        candidates = all_items[mask]
        users = torch.full_like(candidates, user_idx)
        scores = model.predict_proba(users, candidates)
        true_position = torch.nonzero(candidates == true_item, as_tuple=False)
        if true_position.numel() == 0:
            raise ValueError(f"held-out item {true_item} was removed from candidates")
        true_score = scores[int(true_position.item())]
        rank = 1 + int(torch.count_nonzero(scores > true_score))
        if rank <= k:
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 1)

    count = float(len(held_out))
    return {f"hit_rate@{k}": hits / count, f"ndcg@{k}": ndcg / count}
