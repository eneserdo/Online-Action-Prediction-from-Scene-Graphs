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
from torchmetrics.classification import MulticlassF1Score
from utils import classification_scores, conf_mat_from_left_right
from torchmetrics.classification import MulticlassConfusionMatrix

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
        ground_truth_lft_rght = data[-1].y.argmax(dim=1)
        ground_truth = torch.cat([ground_truth_lft_rght[::2], ground_truth_lft_rght[1::2]])
        correct += int((pred == ground_truth).sum())  # Check against ground-truth labels.
        total += pred.size(dim=0)

        if visdom_enabled:
            viz.line(Y=[loss.item()], X=[step], win=loss_window, update="append")
            viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="train_acc")


def test(model, loader):
    model.eval()
    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = [d.to(device) for d in data]

        left_right_pred = model(data)
        # pred = left_right_pred.argmax(dim=1)
        pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:, :3]
        pred = pred_top_3[:, 0]
        ground_truth = data[-1].y.argmax(dim=1)
        pred_top = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                               ground_truth, pred)

        conf_mat_top += conf_mat_from_left_right(ground_truth, pred, action_indices)
        conf_mat_top_3 += conf_mat_from_left_right(ground_truth, pred_top, action_indices)

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
        ground_truth_lft_rght = data[-1].y.argmax(dim=1)
        ground_truth = torch.cat([ground_truth_lft_rght[::2], ground_truth_lft_rght[1::2]])
        correct += int((pred == ground_truth).sum())  # Check against ground-truth labels.
        total += pred.size(dim=0)

    print("Val acc: ", correct / total)
    if visdom_enabled:
        viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="val_acc")


def run_one_epoch(mode, model, loader):
    global step

    if mode in ["test", "val"]:
        model.eval()
    elif mode== "train":
        model.train()

    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    f1_calc_top1=MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro")
    f1_calc_top3=MulticlassF1Score(num_classes=len(ACTIONS), top_k=3, average="macro")
    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS))
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS))

    for data in loader:
        step += 1

        if mode== "train":
            optimizer.zero_grad()
        data = [d.to(device) for d in data]

        left_right_pred = model(data)
        ground_truth_left_right=torch.cat([data[-1].y[::2], data[-1].y[1::2]])

        loss = criterion(left_right_pred, ground_truth_left_right)

        if mode== "train":
            loss.backward()
            optimizer.step()


        # testden gelen
        pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:, :3]
        pred_top_1 = pred_top_3[:, 0]
        ground_truth = ground_truth_left_right.argmax(dim=1)
        pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                               ground_truth, pred_top_1)

        # conf_mat_top += conf_mat_from_left_right(ground_truth, pred_top_1, action_indices)
        # conf_mat_top_3 += conf_mat_from_left_right(ground_truth, pred_top, action_indices)

        # conf_mat_top += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_1.cpu().detach().numpy(), labels=action_indices)
        # conf_mat_top_3 += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_3_final.cpu().detach().numpy(), labels=action_indices)

        # torchmetric

        step_f1_top1=f1_calc_top1(left_right_pred, ground_truth)
        step_f1_top3=f1_calc_top3(left_right_pred, ground_truth)

        step_conf_mat_top1=conf_mat_top1(pred_top_1, ground_truth)
        step_conf_mat_top3=conf_mat_top3(pred_top_3_final, ground_truth)

        

    overall_f1_top1=f1_calc_top1.compute()
    overall_f1_top3=f1_calc_top3.compute()
    
    overall_conf_mat_top1=conf_mat_top1.compute()
    overall_conf_mat_top3=conf_mat_top3.compute()

    # scores = classification_scores(conf_mat_top)
    # scores_top_3 = classification_scores(conf_mat_top_3)

        # if visdom_enabled:
        #     viz.line(Y=[loss.item()], X=[step], win=loss_window, update="append")
        #     viz.line(Y=[scores["F1"]["avg"]["macro"]], X=[step], win=acc_window, update="append", name=mode)
            # viz.line(Y=[scores_top_3], X=[step], win=acc_window, update="append", name=mode)
        
    return overall_f1_top1, overall_f1_top3

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
    start= 1 if args.restore is not None else args.restore
    
    print("Start training...")
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()
        train(model, train_loader)
        stop1 = time.perf_counter()
        # train_acc = test(model, train_loader) # FIXME: Bu şekilde yapmak inanılmaz  inefficient
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        validate(model, val_loader)
        run_one_epoch(model, val_loader)

        # Save model
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                       opj("saved", f"checkpoint_epoch{epoch:0>4}.pt"))

        print(f'#{epoch:03d}: Time {(stop1 - start)//60}:{(stop1 - start)%60}m, Validation: top1-f1: {top1_f1}, top3-f1: {top3_f1}')

    test(model, test_loader)
    # test_acc = test(test_loader)
