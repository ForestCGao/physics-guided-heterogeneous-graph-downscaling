# Copyright (c) 2019 acbull
# Adapted from https://github.com/acbull/pyHGT (MIT).
# See THIRD_PARTY_LICENSES.md for the full upstream notice.
import torch
from torch import nn
from .conv import GeneralConv

class GNN(nn.Module):

    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='dense_hgt', prev_norm=False, last_norm=False, use_RTE=True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm, use_RTE=use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm, use_RTE=use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type, return_all_states=False):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        all_states = {}
        all_attentions = {}
        for (layer_id, gc) in enumerate(self.gcs):
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
            all_states[f'layer_{layer_id}'] = meta_xs.detach()
            all_attentions[f'layer_{layer_id}'] = gc.base_conv.att
        if return_all_states:
            return (meta_xs, all_states, all_attentions)
        else:
            return meta_xs
