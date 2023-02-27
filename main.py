# -*- coding: utf-8 -*-
"""
    TODO:
    - make it fully deterministic (ask arda hoca)
    - implement cProfile
    - weight init appropriately
    - add loss graph
    - add required mechanisms for restored model
"""

import math

import uuid
import matplotlib.pyplot as plt

import os
import torch
import numpy as np

import models
from utils import arg_parser
from dataset import BimanualDataset, get_bimanual_dataset_splits, OBJECTS, RELATIONS, ACTIONS
from torch_geometric.loader import DataLoader
import time
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(0)
np.random.seed(1000)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_RES_PATH = "./res"

def train(model, train_loader):
    model.train()
    loss_track=[]

    for data in train_loader:  
        optimizer.zero_grad()  
        left_pred, right_pred = model(data) 
        loss = criterion(left_pred[0], data[-1].y[::2]) + criterion(right_pred[0], data[-1].y[1::2])
        # data[-1].y means labels of last frames
        # data[-1].y[::2] means even indexed elements (left hands)  
        # data[-1].y[1::2] means odd indexed elements (right hands)  
        loss_track.append(loss)
        loss.backward()
        optimizer.step()

    return loss_track

def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        left_pred, right_pred = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":
    args=arg_parser()
    
    train_set, test_set, val_set = get_bimanual_dataset_splits(root=".", is_processed=False ,test_subject=args.test_subject, validation_take=args.validation_take, temporal_length=args.temporal_length)
    
    train_loader=DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader=DataLoader(test_set, args.batch_size, shuffle=True)
    val_loader=DataLoader(val_set, args.batch_size, shuffle=True)
    
    model = models.GNN(num_edge_feat=len(RELATIONS),
                        conv_in=len(OBJECTS),
                        conv_hidden=args.conv_hidden,
                        conv_out=args.conv_out,
                        batch_size=args.batch_size, 
                        num_action_classes=len(ACTIONS), 
                        hidden_size=args.hidden_size, 
                        num_rnn_layers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 3):
        loss_track=train(model, train_loader)
        # train_acc = test(model, train_loader) # FIXME: Bu şekilde yapmak inanılmaz  inefficient
        # test_acc = test(test_loader)  
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Epoch: {epoch:03d}')