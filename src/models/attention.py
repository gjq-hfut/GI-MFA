"""Attention modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class attention_paper(nn.Module):
    """Attention over 2 item-side representations conditioned on user representations.

    Inputs:
      q: [B, D] user embeddings
      k: [2, D] stacked item embeddings (two sources)
    Output:
      out: [B, D]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = self.query(q)  # [B,D]
        k1 = self.key(k)   # [2,D]
        v = self.value(k)  # [2,D]
        attn = k1 @ q.t()  # [2,B]
        attn = attn / torch.sqrt(torch.tensor(float(self.in_channels), device=attn.device))
        attn = self.softmax(attn)  # softmax over last dim (B); matches notebook behavior
        out = attn.t() @ v  # [B,D]
        return out


class attention_researcher(nn.Module):
    """Self-attention over 2 user-side representations.

    Input:
      input: [2, D]
    Output:
      out: [D]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q = self.query(input).t()  # [D,2]
        k = self.key(input)        # [2,D]
        v = self.value(input)      # [2,D]
        attn = k @ q               # [2,2]
        attn = self.softmax(attn)
        out = attn @ v             # [2,D]
        out = out[0, :] + out[1, :]
        return out
