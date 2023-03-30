import os, sys
import numpy as np
import csv
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import dense_to_sparse

from graph_preprocessing import t_test, define_node_edge

import matplotlib.pyplot as plt
import seaborn as sns


def data_loader(args, train_x, train_label, valid_x, valid_label):
    # make graph topology with only train data
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    train_Node_list = torch.FloatTensor(train_x).to(args.device)
    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)
    train_label = torch.LongTensor(train_label).to(args.device)
    train_dataset = []

    for i, sub in enumerate(train_A_list):
        edge, attr = dense_to_sparse(sub)
        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], edge_index=edge, edge_attr=attr))

    if args.batch_size > len(train_dataset):
        trainLoader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    else:
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))  # full batch

    return trainLoader, validLoader