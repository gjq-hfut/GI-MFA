"""Loss functions.

This repo originally computed BPR loss over a full user-item score matrix.
For scalability, we provide a **mini-batch BPR** implementation that samples
(u, pos, neg) triples and only looks up the corresponding scores.

Assumptions:
- `scores` is a dense tensor of shape [num_users, num_items] **or** [batch_users, num_items]
- `pos_dict[u]` gives a list/1D tensor of positive item indices for user u
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


def _sample_pos(pos_items: Sequence[int]) -> int:
    # pick one positive uniformly
    if len(pos_items) == 1:
        return int(pos_items[0])
    # torch.randint would require a tensor; python is fine here.
    import random
    return int(pos_items[random.randrange(len(pos_items))])


def _sample_neg(exclude: set[int], num_items: int) -> int:
    """Uniformly sample a negative item not in `exclude`."""
    # Simple rejection sampling (fast when positives are sparse).
    import random
    while True:
        j = random.randrange(num_items)
        if j not in exclude:
            return int(j)


def bpr_loss_minibatch(
    scores: torch.Tensor,
    pos_dict: Dict[int, Sequence[int]],
    num_items: int,
    user_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute BPR loss on a mini-batch of users.

    Parameters
    ----------
    scores:
        [num_users, num_items] or [batch_users, num_items]
    pos_dict:
        maps global user id -> iterable of positive item indices
    num_items:
        total number of items (for negative sampling)
    user_ids:
        tensor of global user ids in this batch, shape [B]

    Returns
    -------
    loss: scalar tensor
    """
    device = scores.device
    user_ids = user_ids.to(device)

    pos_idx = []
    neg_idx = []
    batch_row = torch.arange(user_ids.shape[0], device=device)

    # Build sampled indices
    for u in user_ids.tolist():
        pos_items = pos_dict.get(int(u), [])
        if len(pos_items) == 0:
            # skip users without positives; caller should ideally filter them out
            pos = 0
            excl = set()
        else:
            pos = _sample_pos(pos_items)
            excl = set(int(x) for x in pos_items)
        neg = _sample_neg(excl, num_items)
        pos_idx.append(pos)
        neg_idx.append(neg)

    pos_idx_t = torch.tensor(pos_idx, device=device, dtype=torch.long)
    neg_idx_t = torch.tensor(neg_idx, device=device, dtype=torch.long)

    pos_scores = scores[batch_row, pos_idx_t]
    neg_scores = scores[batch_row, neg_idx_t]

    # -log σ(s(u,i+) - s(u,i-))
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss
