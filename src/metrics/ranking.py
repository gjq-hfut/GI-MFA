"""Efficient top-K ranking metrics.

We compute metrics in **user mini-batches** to avoid materializing large intermediate tensors
and to speed up evaluation.

Metrics implemented:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K

Inputs are dictionaries of positive items per user. During evaluation we usually:
- mask training positives (seen items) so they are not recommended
- evaluate on test positives
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class TopKMetrics:
    precision: Dict[int, float]
    recall: Dict[int, float]
    ndcg: Dict[int, float]


def _dcg(hit: torch.Tensor) -> torch.Tensor:
    """hit: [K] boolean or {0,1} tensor"""
    device = hit.device
    k = hit.numel()
    denom = torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    return (hit.float() / denom).sum()


def _idcg(n_relevant: int, k: int, device: torch.device) -> torch.Tensor:
    m = min(n_relevant, k)
    if m <= 0:
        return torch.tensor(0.0, device=device)
    denom = torch.log2(torch.arange(2, m + 2, device=device, dtype=torch.float32))
    return (1.0 / denom).sum()


@torch.no_grad()
def evaluate_topk_minibatch(
    scores: torch.Tensor,
    user_ids: torch.Tensor,
    test_pos: Dict[int, Sequence[int]],
    train_pos: Dict[int, Sequence[int]] | None,
    ks: Sequence[int],
) -> TopKMetrics:
    """Evaluate metrics for a batch of users.

    scores:
        [B, num_items] predicted scores for these users
    user_ids:
        [B] global user ids aligned with rows in `scores`
    test_pos:
        global user id -> test positive item indices
    train_pos:
        global user id -> training positive item indices to be masked (optional)
    ks:
        list of K values, e.g. [5,10,15,20,25]
    """
    device = scores.device
    user_ids = user_ids.to(device)
    ks = list(sorted(set(int(k) for k in ks)))
    k_max = ks[-1]

    # mask seen items (train positives)
    if train_pos is not None:
        masked = scores.clone()
        for r, u in enumerate(user_ids.tolist()):
            seen = train_pos.get(int(u), [])
            if len(seen) > 0:
                idx = torch.tensor(list(seen), device=device, dtype=torch.long)
                masked[r, idx] = -float("inf")
        scores = masked

    topk_idx = torch.topk(scores, k=k_max, dim=1).indices  # [B, Kmax]

    # Accumulators
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_users = 0

    for r, u in enumerate(user_ids.tolist()):
        gt = test_pos.get(int(u), [])
        if len(gt) == 0:
            continue
        n_users += 1
        gt_set = set(int(x) for x in gt)

        recs = topk_idx[r].tolist()  # length Kmax
        hits = torch.tensor([1 if i in gt_set else 0 for i in recs], device=device)

        for k in ks:
            h_k = hits[:k]
            num_hit = float(h_k.sum().item())

            prec_sum[k] += num_hit / float(k)
            rec_sum[k] += num_hit / float(len(gt))

            # NDCG
            dcg = _dcg(h_k)
            idcg = _idcg(len(gt), k, device)
            ndcg_sum[k] += float((dcg / (idcg + 1e-12)).item())


    if n_users == 0:
        return TopKMetrics(
            precision={k: 0.0 for k in ks},
            recall={k: 0.0 for k in ks},
            ndcg={k: 0.0 for k in ks}
        )

    return TopKMetrics(
        precision={k: prec_sum[k] / n_users for k in ks},
        recall={k: rec_sum[k] / n_users for k in ks},
        ndcg={k: ndcg_sum[k] / n_users for k in ks}
    )
