# -*- coding: utf-8 -*-
"""
    TODO:
    - Hyperparameter search
    - log final test results as bar-plot
"""

import glob
import json
import os
import pickle
import random
import time
from datetime import datetime
from os.path import join as opj
import warnings

import functorch
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)

import matplotlib.pyplot as pyplot

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (F1_at_k, arg_parser, calc_metrics_at_once, combine_predictions, get_minutes, metrics_to_str, plot_conf_mat, seed_worker, 
                   visualize_incorrect_preds, create_ground_truth_segments, assign_seg_labels)

from tslearn.metrics import SoftDTWLossPyTorch

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

    f1_calc_top1 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)
    f1_calc_top3 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)  # top_k is intentionally 1

    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)

    # A dictionary that will store the all predictions for all videos. This will be used to calculate f1@k
    # Used -1 as a placeholder value to check if all predictions are filled
    all_predictions_combined = {vid + m_and_hand_status: - np.ones((act.shape[0] - args.temporal_length + 1) * args.temporal_length) for vid, act in dataset_mapping[mode]["vid"].items() for m_and_hand_status in ["_L", "_R", "ML", "MR"]}


    # To ignore warnings about functorch and nan values in confusion matrix
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*', UserWarning)

        for idxes, data in loader:
            step += 1

            if mode == "train":
                optimizer.zero_grad()

            data = [d.to(device) for d in data]

            
            gt_reordered = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data], dim=0)

            batch_size = gt_reordered.shape[1] // 2

            ground_truth_segments = create_ground_truth_segments(torch.argmax(gt_reordered, dim=-1))

            # TODO: Make prediction for each segment using mask
            # 1. from predicted segments
            # or
            # 2. from ground truth segments

            # TODO: obtain 0,1 valued targets of whether action changes via algorithm here or in data processing phase -> FSD 20.09.2023
            # pred_flattened = model(data) # segmentation task -> FSD 20.09.2023
            pred_flattened, segment_change_pred = model(data, ground_truth_segments)  
            # pred_flattened shape: TODO: 

            gt_flattened = torch.flatten(torch.transpose(gt_reordered, 0, 1), start_dim=0, end_dim=1)
            
            if args.sparse_pred:
                pred_flattened, mask = pred_flattened
                gt_flattened = gt_flattened[mask]

            label_loss = label_criterion(pred_flattened, gt_flattened)
            loss = label_loss * label_loss_weight

            if args.seg_loss == "sDTW":
                # NOTE: for label dtw input shapes should be (b,n,l) not (n*b,l) 
                # TODO: 
                # - ablated study (only seg)
                # - gamma val tuned (0.001 etc.)
                # - bceloss distance
                # - ...
                t_shape = segment_change_pred.T.shape + (1,)
                segmentation_loss = segmentation_criterion(segment_change_pred.view(t_shape), ground_truth_segments.view(t_shape)).mean()
            else:
                segmentation_loss = segmentation_criterion(segment_change_pred, ground_truth_segments)
            
            if not args.perfect_seg:
                loss += segmentation_loss

            if mode == "train":
                loss.backward()
                optimizer.step()


            # # # Match predicted segments with ground truth segment labels 
            # This takes 0.006 seconds
            if label_loss_weight == 0:
                # It means that we do not have any label, so we need to assign labels to the segments 
                pred = assign_seg_labels(gt_reordered.cpu(), ground_truth_segments.cpu(), segment_change_pred.cpu().detach())
                pred_flattened_ = torch.flatten(torch.transpose(pred, 0, 1), start_dim=0, end_dim=1)
                pred_flattened = torch.nn.functional.one_hot(pred_flattened_.long(), num_classes=len(ACTIONS)).float()
                pred_flattened = pred_flattened.to(device)
            else:
                # we need to reshatorch
                # TODO: sanity check: assert pred_flattened = assign_seg_labels(...)
                # pred_dummy = assign_seg_labels(gt_reordered.cpu(), ground_truth_segments.cpu(), segment_change_pred.cpu().detach())
                # print("diff between original pred and matched pred: ", np.sum(pred.numpy() != pred_dummy) / (pred_flattened.shape[0]*pred_flattened.shape[1]))
                
                pred_ = pred_flattened.cpu().detach()
                pred = torch.transpose(pred_.unflatten(0, (batch_size*2, args.temporal_length)), 0, 1).argmax(dim=-1)

            # This takes 0.0024 seconds
            combine_predictions(all_predictions_combined, idxes.numpy(), pred, dataset_mapping[mode]["sample"])
            
            

            # # Calculate metrics
            # testden gelen
            pred_top_3 = torch.argsort(pred_flattened, dim=-1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:,0]
            ground_truth = gt_flattened.argmax(dim=-1)
            pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                        ground_truth, pred_top_1)
            # torchmetric
            step_f1_top1 = f1_calc_top1(pred_flattened, ground_truth) # NOTE: Lookup if input can have more than 2 channels
            step_f1_top3 = f1_calc_top3(pred_top_3_final, ground_truth)

            conf_mat_top1(pred_top_1, ground_truth)
            conf_mat_top3(pred_top_3_final, ground_truth)
            # TODO: Log confusion matrices of starting, middle and ending frames predictions.
            # This is so as to understand action segmentation affect.

            if mode != "test":
                wandb.log({f"{mode}/batch/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item(), "Label Loss": label_loss.item(), "Segmentation Loss": segmentation_loss.item()},
                           'batch': step})

            if mode == "test" and args.save_incorrects:
                visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, 
                                          test_idx_sample_mapping=dataset_mapping["test"]["sample"])
            # if (step+1)%20==0:
            #     print("20 BATCH ÇALIŞIYOR SADECE")
            #     break
        
        overall_f1_top1 = f1_calc_top1.compute()
        overall_f1_top3 = f1_calc_top3.compute()

        overall_conf_mat_top1 = conf_mat_top1.compute()
        overall_conf_mat_top3 = conf_mat_top3.compute()

    # TODO: Fix all wandb logs to be consistent with each other. use same dict
    if mode != "test":
        wandb.log({f"{mode}/epoch/": {'top1_F1': overall_f1_top1, 'top3_F1': overall_f1_top3}, 'epoch': epoch})

        
    # it takes 0.659 seconds
    
    seg_save_dir = opj(save_dir, f"test_results_{epoch}") if mode == "test" else None
    
    f1_at_10, f1_at_25, f1_at_50, f1_top1_once = calc_metrics_at_once(all_predictions_combined, 
                                                        dataset_mapping[mode]["vid"], 
                                                        num_classes=len(ACTIONS), 
                                                        temporal_length=args.temporal_length,
                                                        save_dir=seg_save_dir)
    
    wandb.log({f"{mode}/epoch/": {'f1@10': f1_at_10, 'f1@25': f1_at_25, 'f1@50': f1_at_50, "f1_top1_once": f1_top1_once}, 'epoch': epoch})


    # To unify and simplify return statement
    return_dict = {"f1_top1": overall_f1_top1.item(), 
                   "f1_top3": overall_f1_top3.item(), 
                   "cm_top1": overall_conf_mat_top1.cpu(), 
                   "cm_top3": overall_conf_mat_top3.cpu(),
                   "f1@10": f1_at_10,
                   "f1@25": f1_at_25,
                   "f1@50": f1_at_50,
                   "f1_top1_once": f1_top1_once}

    return return_dict


def test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of="f1_top1"):
    
    comparison_metric=choose_best_in_terms_of
    
    weight_list=sorted(glob.glob(opj(save_dir, "*.pt")))
    
    best_metric = 0    
    results = []
    best_i = -1
    for i, w_name in enumerate(weight_list):
        print(f"TESTING {w_name}:")
        checkpoint = torch.load(w_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        returned_dict = run_one_epoch(model, test_loader, 'test', epoch=int((w_name[-7:-3])))
        
        print(f'### [F1] TEST: ', metrics_to_str(returned_dict))

        results.append(returned_dict)        
        if best_metric < returned_dict[comparison_metric]:
            best_metric = returned_dict[comparison_metric]
            best_i = i

    return results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

    # assert args.gpu_id < torch.cuda.device_count(), f"gpu_id:{args.gpu_id} not found. Total gpu count: {torch.cuda.device_count()}"
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    wandb.init(project="rt-action-segmentation-from-scene-graphs", name=args.name, config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online")

    # Use date info to save model weights 
    # save_date=datetime.now().strftime("%m-%d-%Y, %H-%M")

    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, is_processed=not args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               contact_only=args.contact_only,
                                                               fix_fps=args.fix_fps,
                                                               fix_null=args.fix_null)

    train_loader = DataLoader(train_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker)
    
    test_loader = DataLoader(test_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker)

    val_loader = DataLoader(val_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker)

    model = models.ActionRecognition(gnn_type=args.gnn_type,
                                     temporal_type=args.temporal_type,
                                     num_edge_feat=len(RELATIONS),
                                     in_channels=len(OBJECTS),
                                     hidden_channels=args.hidden_channels,
                                     out_channels=args.out_channels,
                                     num_action_classes=len(ACTIONS),
                                     lstm_hidden_size=args.lstm_hidden_size,
                                     heads=args.num_heads,
                                     use_bn=not args.no_bn,
                                     filter_it=args.filter_idle_hold,
                                     use_hand_pooling=not args.use_global_pooling,
                                     edge_dropout=args.edge_dropout,
                                     dropout=args.dropout,
                                     use_cls_token=args.use_cls_token,
                                     sparse_pred=args.sparse_pred,
                                     perfect_seg=args.perfect_seg,
                                     merged_pred=args.merged_pred).to(device=device)

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
    if args.seg_loss == "BCELoss":
        segmentation_criterion = torch.nn.BCELoss(weight=loss_weights).to(device=device)
    elif args.seg_loss == "sDTW":
        # TODO: try different distance measures... cross-entropy??
        segmentation_criterion = SoftDTWLossPyTorch(gamma=args.sdtw_gamma)

    save_dir = f"saved_{args.name}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    # Load saved models if required
    if args.restore is not None:
        assert args.restored_name is not None, "restored_name must be provided if restore is not None"
        checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) # FIXME: It overrides the scheduler. Is it desired?
        print("model and optimizer parameters are restored successfully!")

    
    # dataset_mapping contains mapping for each sample to its location in the videos. 
    # Also, a mapping from video to ground truth action labels
    with open(opj(train_set.processed_dir, "dataset_mapping.pickle"), "rb") as f:
        dataset_mapping = pickle.load(f)
        
    
    # STARTS TRAINING AND VALIDATION
    step = 0
    print("Starts training...")
    start = 1 if args.restore is None else args.restore
    epoch = start
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        train_metrics = run_one_epoch(model, train_loader, 'train', epoch)

        val_metrics = run_one_epoch(model, val_loader, 'val', epoch)


        # Save model
        if epoch % args.saving_freq == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                       opj(save_dir, f"model_epoch{epoch:0>4}.pt"))

        # VALIDATION
        # save confusion matrices
        plt = plot_conf_mat(val_metrics["cm_top1"], f"conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
        wandb.log({'val/confusion_matrix/': {'top1': wandb.Image(plt)}})
        plt3 = plot_conf_mat(val_metrics["cm_top3"], f"conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
        wandb.log({'val/confusion_matrix/': {'top3': wandb.Image(plt3)}})
        plt.close()
        plt3.close()

        scheduler.step()
        stop = time.perf_counter()
        # print(f'### {epoch:03d}: Time {get_minutes(stop - start)}'
        #       f'm ### [F1] Train: top1: {round(train_metrics["f1_top1"], 4)},'
        #       f' top3: {round(train_metrics["f1_top3"], 4)} '
        #       f' top1_once: {round(train_metrics["f1_top1_once"], 4)} '
        #       f'### Val: top1: {round(val_metrics["f1_top1"], 4)}, top3: {round(val_metrics["f1_top3"], 4)} '
        #       f'### [F1@k] Train: k10: {round(train_metrics["f1_at_10"], 4)}, k25: {round(train_metrics["f1_at_25"], 4)}, k50: {round(train_metrics["f1_at_50"], 4)} '
        #       f'### Val: k10: {round(val_metrics["f1_at_10"], 4)}, k25: {round(val_metrics["f1_at_25"], 4)}, k50: {round(val_metrics["f1_at_50"], 4)}')

        print(f'### {epoch:03d}: Time {get_minutes(stop - start)} m ### Train: ', metrics_to_str(train_metrics), ' ### Val: ', metrics_to_str(val_metrics))

    # TEST
    print("Starts test...")
    # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = run_one_epoch(model, test_loader, 'test', epoch=-1)
    # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = test_for_all_saved_weights(model, test_loader)
    test_metrics = test_for_all_saved_weights(model, test_loader)


    plt = plot_conf_mat(test_metrics["cm_top1"], f"TEST_conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top1': wandb.Image(plt)}})
    plt3 = plot_conf_mat(test_metrics["cm_top3"], f"TEST_conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top3': wandb.Image(plt3)}})
    

    print(f'### [F1] TEST: ', metrics_to_str(test_metrics))
    wandb.log({"test_top1": round(test_metrics["f1_top1"], 4), "test_top3": round(test_metrics["f1_top3"], 4), "f1_top1_once": round(test_metrics["f1_top1_once"], 4), "f1_at_10": round(test_metrics["f1_at_10"], 4), "f1_at_25": round(test_metrics["f1_at_25"], 4), "f1_at_50": round(test_metrics["f1_at_50"], 4)})

    plt.close()
    plt3.close()
    torch.cuda.empty_cache()
    wandb.finish()