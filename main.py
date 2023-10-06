# -*- coding: utf-8 -*-
"""
    TODO:
    - Hyperparameter search
    - log final test results as bar-plot
"""

import glob
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

from torch_geometric.data import Batch
from typing import Tuple

import matplotlib.pyplot as pyplot

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (arg_parser, classification_scores, conf_mat_from_left_right,
                   get_minutes, plot_conf_mat, seed_worker, visualize_incorrect_preds, create_ground_truth_segments)

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # To fix the following error:
# OSError: [Errno 24] Too many open files
# Alternatively ulimit can be increased
torch.multiprocessing.set_sharing_strategy('file_system')

# # An attempt to fix the following error:
# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:10.0" after 2627 requests (2627 known processed) with 8 events remaining.
pyplot.ioff()


def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch: int):
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
    f1_calc_top3 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(
        device)  # top_k is intentionally 1

    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*',
                                UserWarning)  # ignores warning about functorch

        for idxes, data in loader:
            step += 1

            if mode == "train":
                optimizer.zero_grad()

            data = [d.to(device) for d in data]

            # ground_truth_left_right = torch.cat([data[-1].y[::2], data[-1].y[1::2]])
            ground_truth_left_right = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data],
                                                  dim=0)  # FIXME: maybe
            # there can be a more efficient way -> FSD 25.09.2023
            ground_truth_segments = create_ground_truth_segments(torch.argmax(ground_truth_left_right, dim=-1))

            # TODO: Make prediction for each segment using mask
            # 1. from predicted segments
            # or
            # 2. from ground truth segments

            # TODO: obtain 0,1 valued targets of whether action changes via algorithm here or in data porcessing phase -> FSD 20.09.2023
            left_right_pred, segment_change_pred = model(
                data, ground_truth_segments)  # left_right_pred = model(data) # segmentation task -> FSD 20.09.2023

            label_loss = label_criterion(left_right_pred, ground_truth_left_right)
            segmentation_loss = segmentation_criterion(segment_change_pred, ground_truth_segments)
            loss = label_loss * label_loss_weight + segmentation_loss

            if mode == "train":
                loss.backward()
                optimizer.step()

            # TODO: Check if working correctly
            ground_truth_left_right_eval = torch.flatten(ground_truth_left_right, start_dim=0, end_dim=1)
            left_right_pred_eval = torch.flatten(left_right_pred, start_dim=0, end_dim=1)

            # testden gelen
            pred_top_3 = torch.argsort(left_right_pred_eval, dim=1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:, 0]
            ground_truth = ground_truth_left_right_eval.argmax(dim=-1)
            pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                           ground_truth, pred_top_1)

            # torchmetric
            step_f1_top1 = f1_calc_top1(left_right_pred_eval, ground_truth)
            step_f1_top3 = f1_calc_top3(pred_top_3_final, ground_truth)

            conf_mat_top1(pred_top_1, ground_truth)
            conf_mat_top3(pred_top_3_final, ground_truth)

            if mode != "test":
                wandb.log({f"{mode}/batch/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item(), "Label Loss": label_loss.item(), "Segmentation Loss": segmentation_loss.item()},
                           'batch': step}, step=step)

            if mode == "test" and args.save_incorrects:
                visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, 
                                          mapping_path="processed10/test_idx_sample_mapping.json")
            # if (step+1)%20==0:
            #     print("20 BATCH ÇALIŞIYOR SADECE")
            #     break


        overall_f1_top1 = f1_calc_top1.compute()
        overall_f1_top3 = f1_calc_top3.compute()

        overall_conf_mat_top1 = conf_mat_top1.compute()
        overall_conf_mat_top3 = conf_mat_top3.compute()

    if mode != "test":
        wandb.log({f"{mode}/epoch/": {'top1_F1': overall_f1_top1, 'top3_F1': overall_f1_top3}, 'epoch': epoch})

    cf1 = wandb.Table(data=overall_conf_mat_top1.cpu().numpy(), columns=ACTIONS, rows=ACTIONS)
    cf3 = wandb.Table(data=overall_conf_mat_top3.cpu().numpy(), columns=ACTIONS, rows=ACTIONS)
    wandb.log({f"{mode}_cf/": {"top1": cf1, "top3": cf3}}, step=step)

    # scores = classification_scores(conf_mat_top)
    # scores_top_3 = classification_scores(conf_mat_top_3)

    wandb.log({'top1/': {'F1': overall_f1_top1}, 'top3/': {'F1': overall_f1_top3}})

    """dummy_i = 0
    if mode == "test":
        dummy_i += 1
        visualize_incorrect_preds(pred_top_1, ground_truth, idxes, 
                                  mapping_path="processed10/test_idx_sample_mapping.json")"""

    if mode in ["test", "val"]:
        return overall_f1_top1, overall_f1_top3, overall_conf_mat_top1, overall_conf_mat_top3
    elif mode == "train":
        return overall_f1_top1, overall_f1_top3


def test_for_all_saved_weights(model, test_loader):
    weight_list=sorted(glob.glob(opj(save_dir, "*.pt")))
    
    best_top1 = 0    
    results = []
    best_i = -1
    for i, w_name in enumerate(weight_list):
        print(f"TESTING {w_name}:")
        checkpoint = torch.load(w_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores_and_conf_mats = run_one_epoch(model, test_loader, 'test', epoch=-1)
        # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = scores_and_conf_mats
        
        print(f"\tTop1: {scores_and_conf_mats[0]}, Top3: {scores_and_conf_mats[1]}")
        
        results.append(scores_and_conf_mats)
        
        if best_top1 < scores_and_conf_mats[0]:
            best_top1 = scores_and_conf_mats[0]
            best_i = i

    return weight_list[best_i], *results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

    # assert args.gpu_id < torch.cuda.device_count(), f"gpu_id:{args.gpu_id} not found. Total gpu count: {torch.cuda.device_count()}"
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    wandb.init(project="online-action-prediction-from-scene-graphs", name=args.name, config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online")

    # Use date info to save model weights 
    # save_date=datetime.now().strftime("%m-%d-%Y, %H-%M")

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, is_processed=not args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               contact_only=args.contact_only,
                                                               fix_fps=args.fix_fps)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,
                              worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,
                             worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=3, drop_last=True,
                            worker_init_fn=seed_worker)

    model = models.ActionRecognition(gnn_type=args.gnn_type,
                                     temporal_type=args.temporal_type,
                                     num_edge_feat=len(RELATIONS),
                                     in_channels=len(OBJECTS),
                                     hidden_channels=args.hidden_channels,
                                     out_channels=args.out_channels,
                                     batch_size=args.batch_size,
                                     num_action_classes=len(ACTIONS),
                                     lstm_hidden_size=args.lstm_hidden_size,
                                     heads=args.num_heads,
                                     use_bn=not args.no_bn,
                                     filter_it=args.filter_idle_hold,
                                     use_hand_pooling=not args.use_global_pooling,
                                     edge_dropout=not args.edge_dropout,
                                     dropout=args.dropout,
                                     use_cls_token=args.use_cls_token).to(device=device)

    print(model)
    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # for mode, run in runs.items():
    wandb.watch(model, log='all')

    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch/*", step_metric="epoch")
    wandb.define_metric("val/epoch/*", step_metric="epoch", summary="max")
    # Note: Currently Wandb does not log images with custom-steps but hopefully in the future it will.
    wandb.define_metric("val/confusion_matrix/*", step_metric="epoch")
    wandb.define_metric("batch")
    wandb.define_metric("train/batch/*", step_metric="batch")
    wandb.define_metric("val/batch/*", step_metric="batch")
    wandb.define_metric("test_top1")
    wandb.define_metric("test_top3")

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = getattr(torch.optim, args.optimizer_type)(model.parameters(), lr=args.learning_rate,
                                                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size,
                                                gamma=args.scheduler_gamma, verbose=True)

    if args.weighted_loss == "none":
        loss_weights = None
    elif args.weighted_loss == "inv":
        loss_weights = 1.0 / torch.tensor(train_set.distribution)
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    elif args.weighted_loss == "sqrt_inv":
        loss_weights = 1.0 / torch.sqrt(torch.tensor(train_set.distribution))
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    elif args.weighted_loss == "effective":
        beta = 0.9999
        effective_num = 1.0 - torch.power(beta, train_set.distribution)
        loss_weights = (1.0 - beta) / effective_num
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    else:
        raise NotImplementedError(f"Weighted loss type {args.weighted_loss} is not implemented")
    
    label_loss_weight = args.label_loss_weight

    # TODO: multiple loss -> FSD 22.09.2023
    label_criterion = torch.nn.CrossEntropyLoss(weight=loss_weights).to(device=device)
    segmentation_criterion = torch.nn.BCELoss(weight=loss_weights).to(device=device)

    save_dir = f"saved_{args.name}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # conf_mat_dir=f"conf_mat_{args.name}"
    # if not os.path.isdir(conf_mat_dir):
    #     os.mkdir(conf_mat_dir)

    # Load saved models if required
    if args.restore is not None:
        assert args.restored_name is not None, "restored_name must be provided if restore is not None"
        checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("model and optimizer parameters are restored successfully!")

    step = 0
    print("Starts training...")
    start = 1 if args.restore is None else args.restore
    epoch = start
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        f_top1_train, f_top3_train = run_one_epoch(model, train_loader, 'train', epoch)

        f_top1_val, f_top3_val, conf_mat_top1_val, conf_mat_top3_val = run_one_epoch(model, val_loader, 'val', epoch)

        # Save model
        if epoch % args.saving_freq == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                       opj(save_dir, f"model_epoch{epoch:0>4}.pt"))

        # save confusion matrices
        plt = plot_conf_mat(conf_mat_top1_val.cpu(), f"conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
        wandb.log({'val/confusion_matrix/': {'top1': wandb.Image(plt)}})
        plt3 = plot_conf_mat(conf_mat_top3_val.cpu(), f"conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
        wandb.log({'val/confusion_matrix/': {'top3': wandb.Image(plt3)}})
        plt.close()
        plt3.close()

        scheduler.step()
        stop = time.perf_counter()
        print(f'### {epoch:03d}: Time {get_minutes(stop - start)}'
              f'm ### [F1] Train: top1: {round(f_top1_train.item(), 4)},'
              f' top3: {round(f_top3_train.item(), 4)} '
              f'### Val: top1: {round(f_top1_val.item(), 4)}, top3: {round(f_top3_val.item(), 4)}')

    # TEST
    print("Starts test...")
    # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = run_one_epoch(model, test_loader, 'test', epoch=-1)
    f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = test_for_all_saved_weights(model, test_loader)

    plt = plot_conf_mat(conf_mat_top1_test.cpu(), f"TEST_conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top1': wandb.Image(plt)}})
    plt3 = plot_conf_mat(conf_mat_top3_test.cpu(), f"TEST_conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top3': wandb.Image(plt3)}})

    print(f'### [F1] TEST: top1: {round(f_top1_test.item(), 4)}, top3: {round(f_top3_test.item(), 4)}')
    # TODO: Log as barplot
    wandb.log({"test_top1": round(f_top1_test.item(), 4), "test_top3": round(f_top3_test.item(), 4)})

    plt.close()
    plt3.close()

    # Old test
    # test(model, test_loader)

    torch.cuda.empty_cache()

    wandb.finish()
