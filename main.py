# -*- coding: utf-8 -*-
"""
    TODO:
    - weight init appropriately
    ~ log conf_matrix
    ~ log batchwise loss/F1 and epochwise loss/F1
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

import matplotlib.pyplot as pyplot


import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (arg_parser, classification_scores, conf_mat_from_left_right,
                   get_minutes, plot_conf_mat, seed_worker)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # An attempt to fix the following error:
# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:10.0" after 2627 requests (2627 known processed) with 8 events remaining.
pyplot.ioff() 

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
    warnings.warn("Temporarily using old test code", UserWarning)
    dummy_i=0

    model.eval()
    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)
    conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*',
                                UserWarning)  # ignores warning about functorch

        for data in loader:  # Iterate in batches over the training/test dataset.
            dummy_i+=1
            data = [d.to(device) for d in data]

            left_right_pred = model(data)
            # pred = left_right_pred.argmax(dim=1)
            pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:, 0]

            ground_truth_left_right = torch.cat([data[-1].y[::2], data[-1].y[1::2]])
            ground_truth = ground_truth_left_right.argmax(dim=1)

            pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                           ground_truth, pred_top_1)

            # conf_mat_top += conf_mat_from_left_right(ground_truth, pred, action_indices)
            # conf_mat_top_3 += conf_mat_from_left_right(ground_truth, pred_top_3_final, action_indices)
            conf_mat_top += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_1.cpu().detach().numpy(),
                                             labels=action_indices)
            conf_mat_top_3 += confusion_matrix(ground_truth.cpu().detach().numpy(),
                                               pred_top_3_final.cpu().detach().numpy(), labels=action_indices)
            # if dummy_i==5:
            #     print("TEST ONLY RUNNED 5 STEP")
            #     break

    scores = classification_scores(conf_mat_top)
    scores_top_3 = classification_scores(conf_mat_top_3)
    print(f"F1 top1: {scores['F1']['avg']['macro']}, top3 {scores_top_3['F1']['avg']['macro']}")

    conf_mat_top = conf_mat_top.astype('float') / conf_mat_top.sum(axis=1)[:, np.newaxis]
    conf_mat_top_3 = conf_mat_top_3.astype('float') / conf_mat_top_3.sum(axis=1)[:, np.newaxis]
    plt = plot_conf_mat(conf_mat_top, "aa_restored_top1", do_return_plot=True)
    plt3 = plot_conf_mat(conf_mat_top_3, "aa_restored_top3", do_return_plot=True)

    wandb.log({"test/confusion_matrix/": {'top1': wandb.Image(plt), 'top3': wandb.Image(plt3)}})
    plt.close()
    plt3.close()
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


def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch:int):
    global step

    if mode in ["test", "val"]:
        model.eval()
    elif mode == "train":
        model.train()

    len_actions = len(ACTIONS)
    action_indices = [range(0, len_actions)]
    # conf_mat_top = np.zeros((len_actions, len_actions), dtype=float)  # old conf_mat
    # conf_mat_top_3 = np.zeros((len_actions, len_actions), dtype=float)

    f1_calc_top1 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)
    f1_calc_top3 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device) # top_k is intentionally 1 
    
    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*',
                                UserWarning)  # ignores warning about functorch

        for data in loader:
            step += 1

            if mode == "train":
                optimizer.zero_grad()
            data = [d.to(device) for d in data]

            left_right_pred = model(data)
            ground_truth_left_right = torch.cat([data[-1].y[::2], data[-1].y[1::2]])

            loss = criterion(left_right_pred, ground_truth_left_right)

            if mode == "train":
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

            # old conf_mat
            # conf_mat_top += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_1.cpu().detach().numpy(), labels=action_indices)
            # conf_mat_top_3 += confusion_matrix(ground_truth.cpu().detach().numpy(), pred_top_3_final.cpu().detach().numpy(), labels=action_indices)

            # torchmetric
            step_f1_top1 = f1_calc_top1(left_right_pred, ground_truth)
            step_f1_top3 = f1_calc_top3(pred_top_3_final, ground_truth)

            step_conf_mat_top1 = conf_mat_top1(pred_top_1, ground_truth)
            step_conf_mat_top3 = conf_mat_top3(pred_top_3_final, ground_truth)

            if mode != "test":
                wandb.log({f"{mode}/batch/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step}, step=step)

            # plt not defined hatası veriyor
            # wandb.log({f'{mode}/confusion_matrix/': {'top1': plt, 'top3': plt3}})

            # if (step+1)%20==0:
            #     print("20 BATCH ÇALIŞIYOR SADECE")
            #     break


    overall_f1_top1 = f1_calc_top1.compute()
    overall_f1_top3 = f1_calc_top3.compute()

    overall_conf_mat_top1 = conf_mat_top1.compute()
    overall_conf_mat_top3 = conf_mat_top3.compute()

    if mode != "test":
        wandb.log({f"{mode}/epoch/": {'top1_F1': overall_f1_top1, 'top3_F1': overall_f1_top3}, 'epoch': epoch})


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

    wandb.init(project="online-action-prediction-from-scene-graphs", name=args.name, config=args.__dict__, mode = "disabled" if args.disable_wandb else "online")

    # Use date info to save model weights 
    # save_date=datetime.now().strftime("%m-%d-%Y, %H-%M")

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, is_processed=not args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               contact_only=args.contact_only)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,  worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,  worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,  worker_init_fn=seed_worker)

    model = models.ActionRecognition(gnn_type=args.gnn_type,
                                     num_edge_feat=len(RELATIONS),
                                     in_channels=len(OBJECTS),
                                     hidden_channels=args.hidden_channels,
                                     out_channels=args.out_channels,
                                     batch_size=args.batch_size,
                                     num_action_classes=len(ACTIONS),
                                     lstm_hidden_size=args.lstm_hidden_size,
                                     heads=args.num_heads,
                                     use_bn=not args.no_bn).to(device=device)

    print(model)
        
    # for mode, run in runs.items():
    wandb.watch(model, log='all')

    # TODO find correct metric names and define them
    # I suppose done...
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch/*", step_metric="epoch")
    wandb.define_metric("val/epoch/*", step_metric="epoch", summary="max")
    wandb.define_metric("batch")
    wandb.define_metric("train/batch/*", step_metric="batch")
    wandb.define_metric("val/batch/*", step_metric="batch")
    wandb.define_metric("test_top1")
    wandb.define_metric("test_top3")
    # wandb.define_metric("F1")
    # wandb.define_metric("Loss")

    # wandb.log({f"{mode}/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}}, step=step)

    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    save_dir=f"saved_{args.name}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # conf_mat_dir=f"conf_mat_{args.name}"
    # if not os.path.isdir(conf_mat_dir):
    #     os.mkdir(conf_mat_dir)

    # Load saved models if required
    if args.restore is not None:
        checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("model and optimizer parameters are restored successfully!")

    step = 0
    print("Start training...")
    start = 1 if args.restore is None else args.restore
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        f_top1_train, f_top3_train = run_one_epoch(model, train_loader, 'train', epoch)

        f_top1_val, f_top3_val, conf_mat_top1_val, conf_mat_top3_val = run_one_epoch(model, val_loader, 'val', epoch)

        # Save model
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()},
                       opj(save_dir, f"model_epoch{epoch:0>4}.pt"))

        # save confusion matrices
        plt = plot_conf_mat(conf_mat_top1_val.cpu(), f"conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
        plt3 = plot_conf_mat(conf_mat_top3_val.cpu(), f"conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
        wandb.log({'val/confusion_matrix/': {'top1': wandb.Image(plt), 'top3': wandb.Image(plt3)}})
        plt.close()
        plt3.close()

        scheduler.step()
        stop = time.perf_counter()
        print(f'### {epoch:03d}: Time {get_minutes(stop - start)}'
              f'm ### [F1] Train: top1: {round(f_top1_train.item(), 4)},'
              f' top3: {round(f_top3_train.item(), 4)} '
              f'### Val: top1: {round(f_top1_val.item(), 4)}, top3: {round(f_top3_val.item(), 4)}')


    # TEST
    f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test=run_one_epoch(model, test_loader, 'test', epoch=-1)
    
    plt = plot_conf_mat(conf_mat_top1_test.cpu(), f"TEST_conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
    plt3 = plot_conf_mat(conf_mat_top3_test.cpu(), f"TEST_conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top1': wandb.Image(plt), 'top3': wandb.Image(plt3)}})
    
    print(f'### [F1] TEST: top1: {round(f_top1_test.item(), 4)}, top3: {round(f_top3_test.item(), 4)}')
    wandb.log({"test_top1": round(f_top1_test.item(), 4), "test_top3": round(f_top3_test.item(), 4)})
    
    plt.close()
    plt3.close()
    
    # Old test
    # test(model, test_loader)

    torch.cuda.empty_cache()

    print("wandb saving...")
    wandb.finish()