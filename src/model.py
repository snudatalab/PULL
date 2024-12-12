"""
Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes functions and structure for the model.
"""

import torch
from torch_geometric.utils import degree
from torch_geometric.nn import  GCNConv
import torch.nn.functional as F
import random

def matmul_divide(z, chunk_size=100):
    """
    Matrix multiplication for the large size graph
    """
    num_chunks = z.size(0) // chunk_size
    prob_adj_parts = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        z_chunk = z[start_idx:end_idx, :]

        prob_adj_part = (z_chunk @ z.t()).sigmoid()
        prob_adj_parts.append(prob_adj_part.detach().cpu())

    if num_chunks * chunk_size < z.size(0):
        z_chunk = z[num_chunks * chunk_size:, :]
        prob_adj_part = (z_chunk @ z.t()).sigmoid()
        prob_adj_parts.append(prob_adj_part.detach().cpu())

    prob_adj_parts = torch.cat(prob_adj_parts, dim=0)
    prob_adj = torch.tensor(prob_adj_parts, device=z.device)
    return prob_adj

def top_k_edges(z, edge_idx, n_edge_add, degree, n_top_node):
    """
    Get top-k edges to add
    """
    top_degree = torch.topk(degree, n_top_node)
    top_degree_node_index = top_degree.indices

    prob_adj = matmul_divide(z)
    prob_adj[edge_idx[0, :], edge_idx[1, :]] = -1
    prob_adj = prob_adj - torch.diag(prob_adj.diag())
    prob_adj[top_degree_node_index, :] += 1

    edge_index = torch.tensor((prob_adj.detach().cpu() > 1).nonzero(as_tuple=False).t(), device=z.device)
    edge_weight = prob_adj[edge_index[0], edge_index[1]]

    # select top-k edge candidates
    edge_weight_topk = torch.topk(edge_weight, int(n_edge_add/2))
    edge_weight_idx = edge_weight_topk.indices
    edge_weight = edge_weight_topk.values - 1
    edge_index = torch.stack((edge_index[0, :][edge_weight_idx], edge_index[1, :][edge_weight_idx]), 0)

    edge_weight = torch.cat([edge_weight, edge_weight])
    edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=-1)

    return edge_index, edge_weight

def split_items(a, n):
    """
    Split the list into n lists
    """
    k, m = divmod(len(a), n)
    return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def top_k_edges_large(z, m_list, n_edge_add, num_chunks=100):
    """
    Get top-k edges to add (for ogb dataset)
    """
    m_chunks = torch.tensor(split_items(m_list.tolist(), num_chunks))
    indexs = []
    topks = []
    for m_indexs in m_chunks:
        m_indexs=m_indexs.to("cuda:"+str(z.get_device()))
        z_chunk = z[m_indexs,:]
        p = (z_chunk @ z.t()).sigmoid()
        new = torch.zeros_like(p).to("cuda:"+str(p.get_device()))
        mask  = p > 0.5
        new[mask] = p[mask]
        new = new.to_sparse()
        edge_weight_topk = torch.topk(new.values(), n_edge_add)
        topk_idx = edge_weight_topk.indices
        topk_values = edge_weight_topk.values
        edge_index = torch.stack((m_indexs[new.indices()[0,:][topk_idx]], new.indices()[1,:][topk_idx]), 0)
        indexs.append(edge_index)
        topks.append(topk_values)

    indexs = torch.cat(tuple(indexs), 1)
    topks = torch.cat(tuple(topks),0)
    final_topk = torch.topk(topks, int(n_edge_add/2))
    edge_index = torch.stack((indexs[0, :][final_topk.indices], indexs[1,:][final_topk.indices]),0)

    edge_weight = torch.cat([final_topk.values, final_topk.values])
    edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=-1)

    return edge_index, edge_weight

class GCNLinkPredictor(torch.nn.Module):
    """
    GCN based Link Predictor. we use 2 layers with 16 units for simplicity
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight=None):
        if edge_weight==None:
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight).relu()
            return self.conv2(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, edge_idx, ratio, epoch):
        n_edge = edge_idx.shape[1]
        n_edge_add = int(n_edge*ratio*(epoch-1))
        # n_edge_add = 0

        # select top-k edge candidates with m
        m = 100
        num_nodes = edge_idx.max().item() + 1
        degrees = degree(edge_idx[0], num_nodes)
        edge_index, edge_weight = top_k_edges(z, edge_idx, n_edge_add, degrees, m)
        # _, m_list = torch.topk(degrees, m, largest=True)
        # edge_index, edge_weight = top_k_edges_large(z, m_list, n_edge_add, num_chunks=10)
        
        return edge_index, edge_weight

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))

        return edge_index, edge_weight

