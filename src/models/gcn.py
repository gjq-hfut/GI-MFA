"""GCN blocks.

The original notebook used simple message passing via adjacency matrix multiplication:
    H^{(k+1)} = A H^{(k)}
and then averaged embeddings across layers.

This implementation supports both dense and sparse adjacency matrices.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _matmul(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


class u_i_GCN(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = int(layers)

    def forward(self, node_embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        all_embeddings = [node_embeddings]
        h = node_embeddings
        for _ in range(self.layers):
            h = _matmul(adj, h)
            all_embeddings.append(h)
        all_embed = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return all_embed


class g_i_GCN(nn.Module):
    def __init__(self, input_dim: int, layers: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.layers = int(layers)

    def forward(self, node_embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        all_embeddings = [node_embeddings]
        h = node_embeddings
        for _ in range(self.layers):
            h = _matmul(adj, h)
            all_embeddings.append(h)
        all_embed = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return all_embed


class g_u_GCN(nn.Module):
    def __init__(self, input_dim: int, layers: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.layers = int(layers)

    def forward(self, node_embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        all_embeddings = [node_embeddings]
        h = node_embeddings
        for _ in range(self.layers):
            h = _matmul(adj, h)
            all_embeddings.append(h)
        all_embed = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return all_embed
