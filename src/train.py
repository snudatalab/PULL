"""
Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes fuctions to train PULL.
"""

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from copy import deepcopy


def train(model, optimizer, train_data, criterion, epoch, z=None):
    """
    train the model.
    :param model: model to use.
    :param optimizer: optimizer for model.
    :param train_data: data to train.
    :param criterion: loss of the model.
    :param epoch: epoch to run.
    :param z: latent variable of edge potential.
    """
    # initialize index and weights of positive edges
    if epoch == 1:
        pos_edge_index = train_data.edge_index
        pos_edge_weight = train_data.edge_label.new_ones(pos_edge_index.shape[1])
    else:
        pos_edge_index_add, pos_edge_weight_add = model.decode_all(z, train_data.edge_index, ratio=0.05,
                                                                   epoch=epoch)
        pos_edge_index, pos_edge_weight = model.merge_edge(train_data.edge_index,
                                                           torch.cat((train_data.edge_label, train_data.edge_label)),
                                                           pos_edge_index_add, pos_edge_weight_add)

        pos_edge_index = pos_edge_index.detach()
        pos_edge_weight = pos_edge_weight.detach()

    # train model
    for inner_epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=int(len(pos_edge_weight)), method='sparse')

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        z = model.encode(train_data.x, pos_edge_index, pos_edge_weight)

        neg_edge_labels = train_data.edge_label.new_zeros(neg_edge_index.size(1))
        edge_label = torch.cat([pos_edge_weight, neg_edge_labels], dim=0)

        out = model.decode(z, edge_label_index).view(-1)

        if epoch == 1:
            loss = criterion(out, edge_label)
        else:
            neg_edge_index_a = negative_sampling(
                edge_index=pos_edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index_a = torch.cat([train_data.edge_label_index, neg_edge_index_a], dim=-1)
            z_a = model.encode(train_data.x, train_data.edge_index, None)

            neg_edge_labels_a = train_data.edge_label.new_zeros(neg_edge_index_a.size(1))
            edge_label_a = torch.cat([train_data.edge_label, neg_edge_labels_a], dim=0)

            out_a = model.decode(z_a, edge_label_index_a).view(-1)
            loss = criterion(out, edge_label) + criterion(out_a, edge_label_a)

        loss.backward()
        optimizer.step()

    return loss, z, pos_edge_index, pos_edge_weight

@torch.no_grad()
def test(data, train_data, model):
    """
    test the model
    :param data: real data for the ground truth
    :param train_data: data which part of edges are removed. used as input for the model.
    :param model: model to predict 
    """
    model.eval()
    z = model.encode(data.x, train_data.edge_index, None)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

