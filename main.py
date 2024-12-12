"""
Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

import os.path as osp
import argparse, random
import numpy as np
import time
import torch_geometric.transforms as T
from torch.nn import BCEWithLogitsLoss
from torch_geometric.datasets import CitationFull, WikipediaNetwork, FacebookPagePage
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE, ARGA, ARGVA
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore")

from src import *

def parse_args():
    """
    parser arguments to run program in cmd
    :return:
    """
    parser = argparse.ArgumentParser()

    # Pre-sets before start the program
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # Hyperparameters for training
    parser.add_argument('--data', type=str, default="Cora_full",
                        help='Cora_full / PubMed_full / chameleon / crocodile / FacebookPagePage')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--verbose', type=str, default="n")

    # Hyperparameters for models
    parser.add_argument('--units', type=int, default=16)
    parser.add_argument('--layers', type=int, default=2)

    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    args = parse_args()

    if args.data in ['Cora_full','PubMed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'CitationFull')
    elif args.data in ["chameleon", "crocodile"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Wikipedia')
    elif args.data == 'FacebookPagePage':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FacebookPagePage')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Set model
    transform = T.Compose([T.NormalizeFeatures(), T.ToDevice(device),
                           T.RandomLinkSplit(num_val=args.val_ratio, num_test=args.test_ratio,
                           is_undirected=True, add_negative_train_samples=False), ])
    # Load data
    if args.data in ['Cora_full', 'PubMed']:
        args.data = args.data.split('_')[0]
        dataset = CitationFull(path, name=args.data, transform=transform)
    elif args.data in ["chameleon", "crocodile"]:
        dataset = WikipediaNetwork(path, name=args.data, transform=transform, geom_gcn_preprocess=False)
    elif args.data == 'FacebookPagePage':
        dataset = FacebookPagePage(path, transform=transform)

    train_dataset, val_dataset, test_dataset = dataset[0]

    if args.verbose=='y':
        print(f"Feature matrix: {train_dataset.x.shape}")
        print(f"Number of training edges: {int(train_dataset.edge_label.sum())}")
        print(f"Number of validation edges: {int(val_dataset.edge_label.sum())}")
        print(f"Number of test edges: {int(test_dataset.edge_label.sum())}\n")

    # Initialize model
    model = GCNLinkPredictor(dataset.num_features, args.units, args.units).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()

    # Train model
    best_val_auc = test_auc = 0
    best_epoch = 0
    z = None
    cnt=0
    val_auc_prev = 0

    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        loss, z, edge_index, edge_weight = train(model, optimizer, train_dataset, criterion, epoch, z)
        val_auc = test(val_dataset, train_dataset, model)
        c_test_auc = test(test_dataset, train_dataset, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            test_auc = test(test_dataset, train_dataset, model)

        else:
            break
        val_auc_prev = val_auc
        if args.verbose == 'y':
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
        
    print(args.data, "|", f'Best Epoch: {best_epoch:02d}, Val: {best_val_auc:.4f}, Test: {test_auc:.4f}')
    return

if __name__ == '__main__':
    main()
