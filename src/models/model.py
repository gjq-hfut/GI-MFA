"""Main model with batch scoring.

This is a cleaned, runnable version of the notebook model with two practical changes:
1) Support sparse adjacency matrices in GCN blocks.
2) Support computing scores for a **subset of users** (mini-batch) to reduce memory.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gcn import u_i_GCN, g_i_GCN, g_u_GCN
from .attention import attention_paper, attention_researcher


class my_model(nn.Module):
    def __init__(self, input_dim: int, group_num: int, user_num: int, paper_num: int, gcn_layers: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.paper_num = int(paper_num)
        self.group_num = int(group_num)
        self.user_num = int(user_num)

        self.embed_dic = self.init_embedding()

        self.GCN1 = u_i_GCN(gcn_layers)
        self.GCN2 = g_u_GCN(input_dim, gcn_layers)
        self.GCN3 = g_i_GCN(input_dim, gcn_layers)

        # Use dynamic feature sizes
        self.fc1 = nn.Linear(self.group_num + self.user_num, self.input_dim, bias=False)
        self.fc2 = nn.Linear(self.group_num + self.user_num, self.input_dim, bias=False)
        self.fc3 = nn.Linear(self.group_num * 2, self.input_dim, bias=False)

        self.item_attn = attention_researcher(self.input_dim)
        self.user_attn = attention_paper(self.input_dim)

    def init_embedding(self) -> nn.ParameterDict:
        initializer = nn.init.xavier_uniform_
        embed_dic = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.input_dim))),
            'paper_emb': nn.Parameter(initializer(torch.empty(self.paper_num, self.input_dim))),
            'group_emb': nn.Parameter(initializer(torch.empty(self.group_num, self.input_dim))),
        })
        return embed_dic

    def forward(
        self,
        train_interaction: torch.Tensor,
        g_u_interaction: torch.Tensor,
        g_p_interaction: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scores.

        If `user_ids` is None: returns full score matrix [U, I]
        If `user_ids` is provided: returns score matrix [B, I] for those users

        Returns
        -------
        scores: [U,I] or [B,I]
        user_feat: [U,D] or [B,D]
        """
        device = self.embed_dic['user_emb'].device

        user_embedding = self.embed_dic['user_emb']
        paper_embedding = self.embed_dic['paper_emb']
        group_embedding = self.embed_dic['group_emb']

        # Build bipartite/hetero node embeddings
        u_p_embedding = torch.cat([user_embedding, paper_embedding], dim=0)       # [U+I, D]
        g_u_embedding = torch.cat([group_embedding, user_embedding], dim=0)       # [G+U, D]
        g_p_embedding = torch.cat([group_embedding, paper_embedding], dim=0)      # [G+I, D]

        # GCN propagation
        u_p_gat_embedding = self.GCN1(u_p_embedding, train_interaction)           # [U+I, D]
        g_u_gat_embedding = self.GCN2(g_u_embedding, g_u_interaction)            # [G+U, D]
        g_p_gat_embedding = self.GCN3(g_p_embedding, g_p_interaction)            # [G+I, D]

        # Split embeddings back
        u_p_u_gat_feature = u_p_gat_embedding[: self.user_num]                   # [U, D]
        u_p_p_gat_feature = u_p_gat_embedding[self.user_num :]                   # [I, D]

        g_u_g_gat_feature = g_u_gat_embedding[: self.group_num]                  # [G, D]
        g_u_u_gat_feature = g_u_gat_embedding[self.group_num :]                  # [U, D]

        g_p_g_gat_feature = g_p_gat_embedding[: self.group_num]                  # [G, D]
        g_p_p_gat_feature = g_p_gat_embedding[self.group_num :]                  # [I, D]

        # Build group-conditioned user representations (per user, aggregate group embeddings)
        # We reuse the original notebook idea: use adjacency to project group embeddings to users.
        # Here we compute:
        #   group_to_user = A_{u,g} @ g_embedding
        # We approximate by multiplying g_u_interaction (G+U x G+U) with node embeddings and slicing.
        # But we already have g_u_gat_embedding, so directly use g_u_u_gat_feature.
        # Combine u-side features from u-p and g-u branches:
        user_feat_all = (u_p_u_gat_feature + g_u_u_gat_feature) / 2.0            # [U, D]

        # Batch selection
        if user_ids is not None:
            user_ids = user_ids.to(device=device, dtype=torch.long)
            user_feat = user_feat_all[user_ids]                                  # [B, D]
        else:
            user_feat = user_feat_all                                            # [U, D]

        # Compute item scores for each paper.
        # We keep the paper-side representations from two branches: u-p and g-p.
        # For each item i:
        #   i_feature = stack([u_p_p_gat_feature[i], g_p_p_gat_feature[i]])  -> [2,D]
        #   paper_out = user_attn(user_feat, i_feature)                    -> [B,D]
        #   score(u,i) = <user_feat[u], paper_out[u]>
        B = user_feat.shape[0]
        scores = torch.empty((B, self.paper_num), device=device, dtype=user_feat.dtype)

        for i in range(self.paper_num):
            i_feature = torch.stack([u_p_p_gat_feature[i], g_p_p_gat_feature[i]], dim=0)  # [2,D]
            paper_out = self.user_attn(user_feat, i_feature)                              # [B,D]
            scores[:, i] = (user_feat * paper_out).sum(dim=1)

        return scores, user_feat
