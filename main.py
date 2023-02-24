# -*- coding: utf-8 -*-
import math

import uuid
import matplotlib.pyplot as plt

import os
import torch
import numpy as np

import models
from utils import arg_parser

import time
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: make it deterministic
random.seed(0)
np.random.seed(1000)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_RES_PATH = "./res"

def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":
    args=arg_parser()
    edge_linear_in=16
    model = models.GNN(edge_linear_in=edge_linear_in,
                       edge_linear_out=args.edge_linear_out,
                       conv_in=args.conv_in,
                       conv_out=args.conv_out,
                       hidden_size=args.hidden_size,
                       batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # for epoch in range(1, 171):
    #     train(model, train_loader)
    #     train_acc = test(model, train_loader)
    #     test_acc = test(test_loader)
    #     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')