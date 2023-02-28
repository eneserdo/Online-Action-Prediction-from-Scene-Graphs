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
# import matplotlib.pyplot as plt

import os
import torch
import numpy as np

import models
from utils import arg_parser
from dataset import BimanualDataset, get_bimanual_dataset_splits, OBJECTS, RELATIONS, ACTIONS
from torch_geometric.loader import DataLoader
import time
import random
from tqdm import tqdm
import visdom

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(34)
np.random.seed(34)
torch.manual_seed(34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_RES_PATH = "./res"

def train(model:torch.nn.Module, train_loader: DataLoader):
    global step
    model.train()

    for data in train_loader:
        step+=1
        optimizer.zero_grad()
        data=[d.to(device) for d in data]  
        # left_pred, right_pred = model(data) 
        left_right_pred = model(data) 
        # loss = criterion(left_pred[0], data[-1].y[::2]) + criterion(right_pred[0], data[-1].y[1::2])
        loss = criterion(left_right_pred[:args.batch_size], data[-1].y[::2]) + criterion(left_right_pred[args.batch_size:], data[-1].y[1::2])
        # data[-1].y means labels of last frames
        # data[-1].y[::2] means even indexed elements (left hands)  
        # data[-1].y[1::2] means odd indexed elements (right hands)  
        loss.backward()
        optimizer.step()
        if visdom_enabled:
            viz.line(Y=[loss.item()],X=[step], win=loss_window, update="append")

def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        left_pred, right_pred = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def validate(model, loader):
    model.eval()
    pass

if __name__ == "__main__":

    args=arg_parser()
    visdom_enabled = args.use_visdom
    
    train_set, test_set, val_set = get_bimanual_dataset_splits(root=".", is_processed=not args.process_it ,test_subject=args.test_subject, validation_take=args.validation_take, temporal_length=args.temporal_length)
    
    train_loader=DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2)
    test_loader=DataLoader(test_set, args.batch_size, shuffle=True, num_workers=2)
    val_loader=DataLoader(val_set, args.batch_size, shuffle=True, num_workers=2)
    
    model = models.GNN(num_edge_feat=len(RELATIONS),
                        conv_in=len(OBJECTS),
                        conv_hidden=args.conv_hidden,
                        conv_out=args.conv_out,
                        batch_size=args.batch_size, 
                        num_action_classes=len(ACTIONS), 
                        hidden_size=args.hidden_size, 
                        num_rnn_layers=1).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    
    
        

    if visdom_enabled:    
        viz = visdom.Visdom() # To make this work run "visdom" in another terminal
        loss_window = viz.line(X=[0], Y=[0], opts=dict(xlabel='Step', ylabel='Loss', title='Training Loss')) 

    step=0
    for epoch in range(1, 3):
        start=time.perf_counter()
        train(model, train_loader)
        stop1=time.perf_counter()
        # train_acc = test(model, train_loader) # FIXME: Bu şekilde yapmak inanılmaz  inefficient
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        # validate(model, val_loader)
        print(f'Epoch: {epoch:03d}')
        print("time: ", stop1-start)
    
    
    # test_acc = test(test_loader)