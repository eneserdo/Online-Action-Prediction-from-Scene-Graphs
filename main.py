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
from os.path import join as opj

import functorch
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

import models
from utils import arg_parser
from dataset import BimanualDataset, get_bimanual_dataset_splits, OBJECTS, RELATIONS, ACTIONS
from torch_geometric.loader import DataLoader
import time
import random
from tqdm import tqdm
import visdom
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(34)
np.random.seed(34)
torch.manual_seed(34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        loss = criterion(left_right_pred[:args.batch_size], data[-1].y[::2]) + criterion(
            left_right_pred[args.batch_size:], data[-1].y[1::2])
        # data[-1].y means labels of last frames
        # data[-1].y[::2] means even indexed elements (left hands)  
        # data[-1].y[1::2] means odd indexed elements (right hands)  
        loss.backward()
        optimizer.step()

        pred = left_right_pred.argmax(dim=1)
        ground_truth = data[-1].y.argmax(dim=1)
        correct = correct + int((pred[:args.batch_size] == ground_truth[::2]).sum()) + int(
            (pred[args.batch_size:] == ground_truth[1::2]).sum())  # Check against ground-truth labels.
        total += pred.size(dim=0)

        if visdom_enabled:
            viz.line(Y=[loss.item()], X=[step], win=loss_window, update="append")
            viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="train_acc")


def test(model, loader):
    model.eval()
    len_actions = len(ACTIONS)
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = [d.to(device) for d in data]

        left_right_pred = model(data)
        # pred = left_right_pred.argmax(dim=1)
        pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:3]
        pred = pred_top_3[0]
        ground_truth = data[-1].y.argmax(dim=1)
        pred_top = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                               ground_truth, pred)

        batch_size = pred.size(dim=0) / 2

        conf_mat_top += conf_mat_from_left_right(ground_truth, pred, batch_size)
        conf_mat_top_3 += conf_mat_from_left_right(ground_truth, pred_top, batch_size)

    scores = classification_scores(conf_mat_top)
    scores_top_3 = classification_scores(conf_mat_top_3)

    print("Test acc: ", scores['accuracy'])
    print("Test top3 acc: ", scores_top_3['accuracy'])

    return scores, scores_top_3


def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:  # Iterate in batches over the training/validation dataset.
        data = [d.to(device) for d in data]

        left_right_pred = model(data)
        pred = left_right_pred.argmax(dim=1)
        ground_truth = data[-1].y.argmax(dim=1)
        correct = correct + int((pred[:args.batch_size] == ground_truth[::2]).sum()) + int(
            (pred[args.batch_size:] == ground_truth[1::2]).sum())  # Check against ground-truth labels.
        total += pred.size(dim=0)

    print("Val acc: ", correct / total)
    if visdom_enabled:
        viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="val_acc")


def conf_mat_from_left_right(truth, pred, batch_size):
    pred_left, pred_right = pred[:batch_size], pred[batch_size:]
    truth_left, truth_right = truth[::2], truth[1::2]
    return confusion_matrix(truth_left.cpu().detach().numpy(), pred_left.cpu().detach().numpy()) + \
           confusion_matrix(truth_right.cpu().detach().numpy(), pred_right.cpu().detach().numpy())


def classification_scores(conf_mat):
    scores = {'recall': {}, 'precision': {}, 'F1': {}}

    TP = np.diagonal(conf_mat)
    TP_p_FN = np.sum(conf_mat, axis=1)  # total class true occurrences
    TP_p_FP = np.sum(conf_mat, axis=0)  # total class predicted occurrences

    mask1 = (TP_p_FN != 0)
    # recall = TP / (TP + FN)
    scores['recall']['all'] = TP[mask1] / TP_p_FN[mask1]

    mask2 = (TP_p_FP != 0)
    # precision = TP / (TP + FP)
    scores['precision']['all'] = TP[mask2] / TP_p_FP[mask2]

    #  mask??

    # F1 = 2 * TP / (2 * TP + FN + FP)
    scores['F1']['all'] = 2 * TP / (TP_p_FP + TP_p_FN)

    # Micro average
    sum_labels = np.sum(TP_p_FN)
    accuracy = np.sum(TP) / sum_labels
    scores['accuracy'] = accuracy
    scores['recall']['avg']['micro'] = accuracy
    scores['precision']['avg']['micro'] = accuracy
    scores['F1']['avg']['micro'] = accuracy

    # Macro average
    n_labels = len(TP)
    scores['recall']['avg']['macro'] = np.sum(scores['recall']['all']) / n_labels
    scores['precision']['avg']['macro'] = np.sum(scores['precision']['all']) / n_labels
    scores['F1']['avg']['macro'] = np.sum(scores['F1']['all']) / n_labels

    # Weighted average
    scores['recall']['avg']['weighted'] = np.sum(scores['recall']['all'] * TP_p_FN) / sum_labels
    scores['precision']['avg']['weighted'] = np.sum(scores['precision']['all'] * TP_p_FN) / sum_labels
    scores['F1']['avg']['weighted'] = np.sum(scores['F1']['all'] * TP_p_FN) / sum_labels

    return scores


if __name__ == "__main__":

    args = arg_parser()

    # Use date info to save model weights 
    # save_date=datetime.now().strftime("%m-%d-%Y, %H-%M")

    visdom_enabled = args.use_visdom

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, is_processed=not args.process_it,
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

    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.isdir("saved"):
        os.mkdir("saved")

    # Load saved models if required
    if args.restore is not None:
        checkpoint = torch.load(opj("saved", f"checkpoint_epoch{args.restore:0>4}.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model and optimizer parameters are restored successfully!")

    if visdom_enabled:
        viz = visdom.Visdom()  # To make this work run "visdom" in another terminal
        loss_window = viz.line(X=[0], Y=[0], opts=dict(xlabel='Step', ylabel='Loss', title='Training Loss'))
        acc_window = viz.line(X=[0], Y=[0], opts=dict(xlabel='Step', ylabel='Accuracy', title='Accuracy'))

    step = 0
    for epoch in range(1, args.max_epochs):
        print(f'Epoch: {epoch:03d}')
        start = time.perf_counter()
        train(model, train_loader)
        stop1 = time.perf_counter()
        # train_acc = test(model, train_loader) # FIXME: Bu şekilde yapmak inanılmaz  inefficient
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        validate(model, val_loader)

        # Save model
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                       opj("saved", f"checkpoint_epoch{epoch:0>4}.pt"))

        print("time: ", stop1 - start)

    test(model, test_loader)
    # test_acc = test(test_loader)
