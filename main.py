# -*- coding: utf-8 -*-
"""
    TODO:
    - make it fully deterministic (ask arda hoca)
    - implement cProfile
    - weight init appropriately
    - add loss graph
    - add required mechanisms for restored model
    - calculate num of parameter in the model
"""

import math
# import matplotlib.pyplot as plt

import os
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

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


def train(model: torch.nn.Module, train_loader: DataLoader):
    global step
    model.train()
    correct = 0
    total = 0

    for data in train_loader:
        step += 1
        optimizer.zero_grad()
        data = [d.to(device) for d in data]
        # left_pred, right_pred = model(data) 
        left_right_pred = model(data)
        # loss = criterion(left_pred[0], data[-1].y[::2]) + criterion(right_pred[0], data[-1].y[1::2])
        loss = criterion(left_right_pred[:args.batch_size], data[-1].y[::2]) + criterion(left_right_pred[args.batch_size:], data[-1].y[1::2])
        # data[-1].y means labels of last frames
        # data[-1].y[::2] means even indexed elements (left hands)  
        # data[-1].y[1::2] means odd indexed elements (right hands)  
        loss.backward()
        optimizer.step()

        pred = left_right_pred.argmax(dim=1)
        ground_truth = data[-1].y.argmax(dim=1)
        correct = correct + int((pred[:args.batch_size] == ground_truth[::2]).sum()) + int((pred[args.batch_size:] == ground_truth[1::2]).sum())  # Check against ground-truth labels.
        total = total + 2 * args.batch_size  # prone to error due to last epoch's unexpected batch_size

        if visdom_enabled:
            viz.line(Y=[loss.item()], X=[step], win=loss_window, update="append")
            viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="train_acc")


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = [d.to(device) for d in data]

        left_right_pred = model(data)
        pred = left_right_pred.argmax(dim=1)
        ground_truth = data[-1].y.argmax(dim=1)
        correct = correct + int((pred[:args.batch_size] == ground_truth[::2]).sum()) + int(
            (pred[args.batch_size:] == ground_truth[1::2]).sum())  # Check against ground-truth labels.
        total = total + 2 * args.batch_size  # prone to error due to last epoch's unexpected batch_size

    if visdom_enabled:
        viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="test_acc")
    # return correct / len(loader.dataset)  # Derive ratio of correct predictions.


#  Classification macro F1 score
def macro_f1_score(output, target):
    # Convert output and target to numpy arrays
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    # Calculate F1 score for each class
    f1_scores = f1_score(target, output, average=None)

    # Calculate macro F1 score
    macro_f1 = torch.tensor(f1_scores).mean()

    return macro_f1

# Normalized confusion matrix
def norm_conf_mat(output, target):
    # Convert output and target to numpy arrays
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    return confusion_matrix(output, target, normalize='true')


def validate(model, loader):
    model.eval()
    pass


if __name__ == "__main__":

    args = arg_parser()
    visdom_enabled = args.use_visdom

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=".", is_processed=not args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = models.GNN(num_edge_feat=len(RELATIONS),
                       conv_in=len(OBJECTS),
                       conv_hidden=args.conv_hidden,
                       conv_out=args.conv_out,
                       batch_size=args.batch_size,
                       num_action_classes=len(ACTIONS),
                       hidden_size=args.hidden_size,
                       num_rnn_layers=1).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if visdom_enabled:
        viz = visdom.Visdom()  # To make this work run "visdom" in another terminal
        loss_window = viz.line(X=[0], Y=[0], opts=dict(xlabel='Step', ylabel='Loss', title='Training Loss'))
        acc_window = viz.line(X=[0], Y=[0], opts=dict(xlabel='Step', ylabel='Accuracy', title='Accuracy'))

    step = 0
    for epoch in range(1, 5):
        print(f'Epoch: {epoch:03d}')
        start = time.perf_counter()
        train(model, train_loader)
        stop1 = time.perf_counter()
        # train_acc = test(model, train_loader) # FIXME: Bu şekilde yapmak inanılmaz  inefficient
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        validate(model, val_loader)
        
        print("time: ", stop1 - start)

    test(model, test_loader)
    # test_acc = test(test_loader)
