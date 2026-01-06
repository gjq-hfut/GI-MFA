"""Training entrypoint with runnable `load_dataset` and per-batch scoring.

Data format :
- user_item_csv: interactions between user and paper/item
- group_user_csv: membership between group and user
- group_item_csv: membership between group and paper/item

The loader is designed to be robust to column names. It will try common names first,
otherwise it falls back to the first two columns.

Outputs:
- Sparse normalized adjacency matrices for the three graphs
- train/test positive dicts per user
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .training.losses import bpr_loss_minibatch
from .metrics.ranking import evaluate_topk_minibatch
from .models.model import my_model


@dataclass
class TrainConfig:
    seed: int = 20
    device: str = "cuda"
    lr: float = 1e-3
    num_epochs: int = 200
    batch_size: int = 256
    eval_batch_size: int = 512
    topks: Sequence[int] = (5, 10, 15, 20, 25)
    eval_ratio: float = 0.2

    model: dict | None = None
    data: dict | None = None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_user_batches(user_ids: torch.Tensor, batch_size: int):
    for i in range(0, user_ids.numel(), batch_size):
        yield user_ids[i : i + batch_size]


def _pick_cols(df: pd.DataFrame, candidates_a: Sequence[str], candidates_b: Sequence[str]) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    a = next((col_map[c] for c in candidates_a if c in cols), None)
    b = next((col_map[c] for c in candidates_b if c in cols), None)
    if a is not None and b is not None:
        return a, b
    # fallback: first two columns
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns.")
    return df.columns[0], df.columns[1]


def _remap_ids(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series, Dict[int, int], Dict[int, int]]:
    a_uni = pd.Index(a.unique())
    b_uni = pd.Index(b.unique())
    a_map = {int(v): i for i, v in enumerate(a_uni)}
    b_map = {int(v): i for i, v in enumerate(b_uni)}
    a_new = a.map(lambda x: a_map[int(x)]).astype(int)
    b_new = b.map(lambda x: b_map[int(x)]).astype(int)
    return a_new, b_new, a_map, b_map


def _build_bipartite_adj(num_left: int, num_right: int, edges_left: np.ndarray, edges_right: np.ndarray) -> torch.Tensor:
    """Return symmetric normalized adjacency for a bipartite graph.

    Node order: [left nodes..., right nodes...], total N = L + R

    We build edges:
      (u, L + i) and (L + i, u)
    and apply D^{-1/2} A D^{-1/2}.
    """
    N = num_left + num_right
    row = np.concatenate([edges_left, num_left + edges_right])
    col = np.concatenate([num_left + edges_right, edges_left])

    # add self-loops
    self_idx = np.arange(N)
    row = np.concatenate([row, self_idx])
    col = np.concatenate([col, self_idx])

    data = np.ones_like(row, dtype=np.float32)

    idx = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    val = torch.tensor(data, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, size=(N, N))
    A = A.coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    r, c = A.indices()
    v = A.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    return torch.sparse_coo_tensor(A.indices(), v, size=A.size()).coalesce()


def _split_train_test_by_user(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    eval_ratio: float,
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    rng = np.random.default_rng(seed)
    train_pos: Dict[int, List[int]] = {}
    test_pos: Dict[int, List[int]] = {}

    # group interactions by user
    from collections import defaultdict
    bucket = defaultdict(list)
    for u, i in zip(user_ids, item_ids):
        bucket[int(u)].append(int(i))

    for u, items in bucket.items():
        items = list(dict.fromkeys(items))  # unique preserve order
        if len(items) == 1:
            train_pos[u] = items
            test_pos[u] = []
            continue
        n_test = max(1, int(round(len(items) * eval_ratio)))
        test_items = set(rng.choice(items, size=min(n_test, len(items)), replace=False).tolist())
        train_items = [i for i in items if i not in test_items]
        if len(train_items) == 0:
            # ensure at least one train
            train_items = [items[0]]
            test_items.discard(items[0])
        train_pos[u] = train_items
        test_pos[u] = [i for i in items if i in test_items]
    return train_pos, test_pos


def load_dataset(cfg: TrainConfig):
    data_cfg = cfg.data or {}
    user_item_path = Path(data_cfg.get("user_item_csv", "data/raw/user_item.csv"))
    group_user_path = Path(data_cfg.get("group_user_csv", "data/raw/group_user.csv"))
    group_item_path = Path(data_cfg.get("group_item_csv", "data/raw/group_item.csv"))

    ui = pd.read_csv(user_item_path)
    gu = pd.read_csv(group_user_path)
    gi = pd.read_csv(group_item_path)

    # infer columns
    u_col, i_col = _pick_cols(ui, ["user", "user_id", "uid", "u"], ["paper", "item", "item_id", "iid", "i"])
    g_col1, u_col2 = _pick_cols(gu, ["group", "group_id", "gid", "g"], ["user", "user_id", "uid", "u"])
    g_col2, i_col2 = _pick_cols(gi, ["group", "group_id", "gid", "g"], ["paper", "item", "item_id", "iid", "i"])

    # remap to contiguous ids (we allow inconsistent raw ids across files)
    ui_u = ui[u_col].astype(int)
    ui_i = ui[i_col].astype(int)
    # build unified user mapping based on user_item + group_user
    all_users = pd.Index(pd.concat([ui_u, gu[u_col2].astype(int)], axis=0).unique())
    user_map = {int(v): idx for idx, v in enumerate(all_users)}
    # unified item mapping based on user_item + group_item
    all_items = pd.Index(pd.concat([ui_i, gi[i_col2].astype(int)], axis=0).unique())
    item_map = {int(v): idx for idx, v in enumerate(all_items)}
    # unified group mapping based on group_user + group_item
    all_groups = pd.Index(pd.concat([gu[g_col1].astype(int), gi[g_col2].astype(int)], axis=0).unique())
    group_map = {int(v): idx for idx, v in enumerate(all_groups)}

    ui_u_idx = ui_u.map(lambda x: user_map[int(x)]).to_numpy(dtype=np.int64)
    ui_i_idx = ui_i.map(lambda x: item_map[int(x)]).to_numpy(dtype=np.int64)
    gu_g_idx = gu[g_col1].astype(int).map(lambda x: group_map[int(x)]).to_numpy(dtype=np.int64)
    gu_u_idx = gu[u_col2].astype(int).map(lambda x: user_map[int(x)]).to_numpy(dtype=np.int64)
    gi_g_idx = gi[g_col2].astype(int).map(lambda x: group_map[int(x)]).to_numpy(dtype=np.int64)
    gi_i_idx = gi[i_col2].astype(int).map(lambda x: item_map[int(x)]).to_numpy(dtype=np.int64)

    num_users = len(user_map)
    num_items = len(item_map)
    num_groups = len(group_map)

    # split train/test from user-item interactions
    train_pos, test_pos = _split_train_test_by_user(ui_u_idx, ui_i_idx, cfg.eval_ratio, cfg.seed)

    # adjacency matrices
    train_interaction = _build_bipartite_adj(num_users, num_items, ui_u_idx, ui_i_idx)
    g_u_interaction = _build_bipartite_adj(num_groups, num_users, gu_g_idx, gu_u_idx)
    g_p_interaction = _build_bipartite_adj(num_groups, num_items, gi_g_idx, gi_i_idx)

    model_inputs = dict(
        train_interaction=train_interaction,
        g_u_interaction=g_u_interaction,
        g_p_interaction=g_p_interaction,
    )

    # also return counts to fill cfg.model if not set
    if cfg.model is None:
        cfg.model = {}
    cfg.model.setdefault("group_num", num_groups)
    cfg.model.setdefault("user_num", num_users)
    cfg.model.setdefault("paper_num", num_items)

    return num_users, num_items, train_pos, test_pos, model_inputs


def parse_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(
        seed=int(raw.get("seed", 20)),
        device=str(raw.get("device", "cuda")),
        lr=float(raw.get("lr", 1e-3)),
        num_epochs=int(raw.get("num_epochs", 200)),
        batch_size=int(raw.get("batch_size", 256)),
        eval_batch_size=int(raw.get("eval_batch_size", 512)),
        topks=tuple(raw.get("topks", [5, 10, 15, 20, 25])),
        eval_ratio=float(raw.get("eval_ratio", 0.2)),
        model=raw.get("model"),
        data=raw.get("data"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    num_users, num_items, train_pos, test_pos, model_inputs = load_dataset(cfg)

    # Move adjacency to device (sparse)
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(device)

    # Fill model args
    model_args = cfg.model or {}
    model_args.setdefault("input_dim", 128)
    model_args.setdefault("group_num", model_inputs["g_u_interaction"].size(0) - num_users)  # G
    model_args.setdefault("user_num", num_users)
    model_args.setdefault("paper_num", num_items)

    model = my_model(**model_args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    all_users = torch.arange(num_users, dtype=torch.long)

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_users in iter_user_batches(all_users, cfg.batch_size):
            batch_users = batch_users.to(device)
            optim.zero_grad()

            # Per-batch scoring (NO full score matrix)
            scores, _ = model(user_ids=batch_users, **model_inputs)  # [B, I]

            loss = bpr_loss_minibatch(
                scores=scores,
                pos_dict=train_pos,
                num_items=num_items,
                user_ids=batch_users,
            )
            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        # Evaluation (batched over users; still per-batch scoring)
        model.eval()
        metric_acc = {k: {"p": 0.0, "r": 0.0, "n": 0.0} for k in cfg.topks}
        n_batches_eval = 0

        with torch.no_grad():
            for batch_users in iter_user_batches(all_users, cfg.eval_batch_size):
                batch_users = batch_users.to(device)
                scores, _ = model(user_ids=batch_users, **model_inputs)  # [B,I]
                mets = evaluate_topk_minibatch(
                    scores=scores,
                    user_ids=batch_users,
                    test_pos=test_pos,
                    train_pos=train_pos,
                    ks=cfg.topks,
                )
                for k in cfg.topks:
                    metric_acc[k]["p"] += mets.precision[k]
                    metric_acc[k]["r"] += mets.recall[k]
                    metric_acc[k]["n"] += mets.ndcg[k]
                n_batches_eval += 1

        report = []
        for k in cfg.topks:
            report.append(
                f"@{k} P={metric_acc[k]['p']/max(n_batches_eval,1):.4f} "
                f"R={metric_acc[k]['r']/max(n_batches_eval,1):.4f} "
                f"NDCG={metric_acc[k]['n']/max(n_batches_eval,1):.4f} "
            )
        print(f"epoch {epoch} | loss {epoch_loss/max(n_batches,1):.4f} | " + " | ".join(report))


if __name__ == "__main__":
    main()
