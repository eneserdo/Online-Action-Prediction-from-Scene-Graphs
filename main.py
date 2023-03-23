# -*- coding: utf-8 -*-
"""
    TODO:
    - make it fully deterministic (ask arda hoca)
    - weight init appropriately
    - torchmetric top_k=3 not working
    - log conf_matrix
    - log batchwise loss/F1 and epochwise loss/F1
"""

import os
import random
import time
from datetime import datetime
from os.path import join as opj
import warnings

import functorch
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (arg_parser, classification_scores, conf_mat_from_left_right,
                   get_minutes, save_conf_mat)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(34)
np.random.seed(34)
torch.manual_seed(34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# def train(model: torch.nn.Module, train_loader: DataLoader):
#     global step
#     model.train()
#     correct = 0
#     total = 0

#     for data in train_loader:
#         step += 1
#         optimizer.zero_grad()
#         data = [d.to(device) for d in data]
#         # left_pred, right_pred = model(data) 
#         left_right_pred = model(data)
#         # loss = criterion(left_pred[0], data[-1].y[::2]) + criterion(right_pred[0], data[-1].y[1::2])
#         loss = criterion(left_right_pred[:args.batch_size], data[-1].y[::2]) + criterion(
#             left_right_pred[args.batch_size:], data[-1].y[1::2])
#         # data[-1].y means labels of last frames
#         # data[-1].y[::2] means even indexed elements (left hands)  
#         # data[-1].y[1::2] means odd indexed elements (right hands)  
#         loss.backward()
#         optimizer.step()

#         pred = left_right_pred.argmax(dim=1)
#         ground_truth_lft_rght = data[-1].y.argmax(dim=1)
#         ground_truth = torch.cat([ground_truth_lft_rght[::2], ground_truth_lft_rght[1::2]])
#         correct += int((pred == ground_truth).sum())  # Check against ground-truth labels.
#         total += pred.size(dim=0)

#         if visdom_enabled:
#             viz.line(Y=[loss.item()], X=[step], win=loss_window, update="append")
#             viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="train_acc")


def test(model, loader):
    model.eval()
    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*', UserWarning) # ignores warning about functorch

        for data in loader:  # Iterate in batches over the training/test dataset.
            data = [d.to(device) for d in data]

            left_right_pred = model(data)
            # pred = left_right_pred.argmax(dim=1)
            pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:, 0]

            ground_truth_left_right=torch.cat([data[-1].y[::2], data[-1].y[1::2]])
            ground_truth = ground_truth_left_right.argmax(dim=1)

                
            pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                ground_truth, pred_top_1)

            # conf_mat_top += conf_mat_from_left_right(ground_truth, pred, action_indices)
            # conf_mat_top_3 += conf_mat_from_left_right(ground_truth, pred_top_3_final, action_indices)
            conf_mat_top += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_1.cpu().detach().numpy(), labels=action_indices)
            conf_mat_top_3 += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_3_final.cpu().detach().numpy(), labels=action_indices)


    scores = classification_scores(conf_mat_top)
    scores_top_3 = classification_scores(conf_mat_top_3)
    print(f"F1 top1: {scores['F1']['avg']['macro']}, top3 {scores_top_3['F1']['avg']['macro']}")

    conf_mat_top = conf_mat_top.astype('float') / conf_mat_top.sum(axis=1)[:, np.newaxis]
    conf_mat_top_3 = conf_mat_top_3.astype('float') / conf_mat_top_3.sum(axis=1)[:, np.newaxis]
    save_conf_mat(conf_mat_top, "aa_resoterd_top1")
    save_conf_mat(conf_mat_top_3, "aa_resoterd_top3")



#     print("Test acc: ", scores['accuracy'])
#     print("Test top3 acc: ", scores_top_3['accuracy'])

#     return scores, scores_top_3


# def validate(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     for data in loader:  # Iterate in batches over the training/validation dataset.
#         data = [d.to(device) for d in data]

#         left_right_pred = model(data)
#         pred = left_right_pred.argmax(dim=1)
#         ground_truth_lft_rght = data[-1].y.argmax(dim=1)
#         ground_truth = torch.cat([ground_truth_lft_rght[::2], ground_truth_lft_rght[1::2]])
#         correct += int((pred == ground_truth).sum())  # Check against ground-truth labels.
#         total += pred.size(dim=0)

#     print("Val acc: ", correct / total)
#     if visdom_enabled:
#         viz.line(Y=[correct / total], X=[step], win=acc_window, update="append", name="val_acc")


def run_one_epoch(model:torch.nn.Module, loader: DataLoader, mode: str):
    global step

    if mode in ["test", "val"]:
        model.eval()
    elif mode == "train":
        model.train()    

    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    f1_calc_top1=MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)
    f1_calc_top3=MulticlassF1Score(num_classes=len(ACTIONS), top_k=3, average="macro").to(device)
    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*', UserWarning) # ignores warning about functorch

        for data in loader:
            step += 1

            if mode == "train":
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


            wandb.log({f"{mode}/":{'top1_F1':step_f1_top1, 'top3_F1':step_f1_top3, "Loss": loss.item()}}, step=step)
            # if (step+1)%20==0:
            #     print("20 BATCH ÇALIŞIYOR SADECE")
            #     break

        

    overall_f1_top1=f1_calc_top1.compute()
    overall_f1_top3=f1_calc_top3.compute()
    
    overall_conf_mat_top1=conf_mat_top1.compute()
    overall_conf_mat_top3=conf_mat_top3.compute()

    

    # cf1=wandb.Table(data=overall_conf_mat_top1.cpu().numpy(), columns=ACTIONS, rows=ACTIONS)
    # cf3=wandb.Table(data=overall_conf_mat_top3.cpu().numpy(), columns=ACTIONS, rows=ACTIONS)
    # wandb.log({f"{mode}_cf/":{"top1": cf1, "top3": cf3}}, step=step)

    # scores = classification_scores(conf_mat_top)
    # scores_top_3 = classification_scores(conf_mat_top_3)


    # wandb.log({'top1/':{'F1':overall_f1_top1}, 'top3/':{'F1':overall_f1_top3}})

    if mode in ["test", "val"]:
        return overall_f1_top1, overall_f1_top3, overall_conf_mat_top1, overall_conf_mat_top3
    elif mode == "train":
        return overall_f1_top1, overall_f1_top3



if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)
    
    wandb.init(project="online-action-prediction-from-scene-graphs",  name=args.name)
    
    # Use date info to save model weights 
    # save_date=datetime.now().strftime("%m-%d-%Y, %H-%M")

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, is_processed=not args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               contact_only=args.contact_only)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = models.ActionRegonition(num_edge_feat=len(RELATIONS),
                                    conv_in=len(OBJECTS),
                                    conv_hidden=args.conv_hidden,
                                    conv_out=args.conv_out,
                                    batch_size=args.batch_size,
                                    num_action_classes=len(ACTIONS),
                                    hidden_size=args.hidden_size,
                                    num_rnn_layers=1).to(device=device)

    #for mode, run in runs.items():
    wandb.watch(model, log='all')
    #runs['job'].define_metric("batch")

    # wandb.define_metric("epoch")
    # wandb.define_metric("F1", step_metric="epoch")
    #wandb.define_metric("batch")
    # wandb.define_metric("F1")
    # wandb.define_metric("Loss")



    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.isdir("saved"):
        os.mkdir("saved")

    if not os.path.isdir("conf_mat"):
        os.mkdir("conf_mat")

    # Load saved models if required
    if args.restore is not None:
        checkpoint = torch.load(opj("saved", f"checkpoint_epoch{args.restore:0>4}.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("model and optimizer parameters are restored successfully!")


    step = 0
    start= 1 if args.restore is None else args.restore
    
    print("Start training...")
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        f_top1_train, f_top3_train=run_one_epoch(model, train_loader, 'train')
        
        f_top1_val, f_top3_val, conf_mat_top1_val, conf_mat_top3_val = run_one_epoch(model, val_loader, 'val')

        # Save model
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()},
                       opj("saved", f"checkpoint_epoch{epoch:0>4}.pt"))


        # save confusion matrices
        save_conf_mat(conf_mat_top1_val.cpu(), f"conf_mat_top1_e{epoch:0>4}")
        save_conf_mat(conf_mat_top3_val.cpu(), f"conf_mat_top3_e{epoch:0>4}")

        scheduler.step()
        stop = time.perf_counter()
        print(f'### {epoch:03d}: Time {get_minutes(stop-start)}m ### [F1] Train: top1: {round(f_top1_train.item(), 4)}, top3: {round(f_top3_train.item(), 4)} ### Val: top1: {round(f_top1_val.item(), 4)}, top3: {round(f_top3_val.item(), 4)}')



    # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test=run_one_epoch(model, test_loader, 'test')

    # print(f'### [F1] TEST: top1: {round(f_top1_test.item(), 4)}, top3: {round(f_top3_test.item(), 4)}')
    
    # save_conf_mat(conf_mat_top1_test.cpu(), f"TEST_conf_mat_top1_e{epoch}")
    # save_conf_mat(conf_mat_top3_test.cpu(), f"TEST_conf_mat_top3_e{epoch}")
    
    print("Temporaly using old test code")
    test(model, test_loader)


    print("wandb saving...")
    wandb.finish()