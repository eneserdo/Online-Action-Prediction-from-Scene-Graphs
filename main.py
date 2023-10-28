# -*- coding: utf-8 -*-
"""
    TODO:
    - Hyperparameter search
    - log final test results as bar-plot
"""

import glob
import json
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
from metrics import f1_at_k_single_example

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (arg_parser, classification_scores, conf_mat_from_left_right,
                   get_minutes, get_weight_loss, plot_conf_mat, seed_worker, visualize_incorrect_preds)
from visualization_utils import plot_segmentation


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


class F1_at_k:
    # Very custom class for calculating f1@k
    def __init__(self, num_classes, save_dir="tests"):
        self.num_classes = num_classes
        self.k_10 = []
        self.k_25 = []
        self.k_50 = []
        self.f_regular = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.save_dir = opj(save_dir, args.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.class_id_to_label = {i: label for i, label in enumerate(ACTIONS)}
        
    def __call__(self, y_true, y_pred, segment_name):
         
        assert len(y_true) == 2 and y_pred.shape[0] == 4, "y_true and y_pred must be a list of 2 elements (left and right hand)"
        
        assert len(y_true[0]) == y_pred.shape[1], "y_true and y_pred must have same length"
        
        plot_save_dir = opj(save_dir, segment_name)

        for i, (y_t, y_p) in enumerate(zip([y_true[0], y_true[1], y_true[1], y_true[0]], y_pred)):
            self.k_10.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.1))
            self.k_25.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.25))
            self.k_50.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.5))

            #TODO: save segmentation plots 
            hand_name= "_L_" if i % 2 == 0 else "_R_"
            mirror_name= "" if i < 2 else "M"
            plot_segmentation(y_t, y_p, class_id_to_label=self.class_id_to_label, save_file=self.plot_save_dir + hand_name + mirror_name + ".png", bar_height=50)
        
            # TODO: Calculate regular f1 score
            # f1_score(torch.tensor(y_pred_L), torch.tensor(y_true_L))

    def compute(self):
        return np.mean(self.k_10), np.mean(self.k_25), np.mean(self.k_50)
    

def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch: int, calc_f1_at_k=False):
    global step

    if mode  == "test":
        model.eval()

        if calc_f1_at_k:
            # idx_sample_mapping: [for each sample] idx -> (subject, task, take, start_frame, end_frame, if_mirrored)
            with open(opj(processed_dir, "test_idx_sample_mapping.json"), "r") as f:
                test_idx_sample_mapping = json.load(f)
            
            # test_ground_truth: [for each video]  video_idx -> (ground_truth_L, ground_truth_R)
            with open(opj(processed_dir, "test_ground_truth.json"), "r") as f:
                test_ground_truth = json.load(f)


            # Video lengthes cumulatively:
            video_lengths = np.cumsum([len(act_arr_R) for _, act_arr_R in test_ground_truth.values()]).tolist()
            f1_at_k = F1_at_k(num_classes=len(ACTIONS))

        
    elif mode == "val":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(f"mode must be one of [train, val, test] but got {mode}")

    len_actions = len(ACTIONS)

    f1_calc_top1 = MulticlassF1Score(num_classes=len_actions, top_k=1, average="macro").to(device)
    f1_calc_top3 = MulticlassF1Score(num_classes=len_actions, top_k=1, average="macro").to(device)  # top_k is intentionally 1

    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len_actions, normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len_actions, normalize="true").to(device)

    if mode == "test" and calc_f1_at_k:
        prediction_video=np.zeros((4, video_lengths[0]))
        test_video_idx=0

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*', UserWarning)  # ignores warning about functorch

        for idxes, data in loader:
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
            # TODO: capsulate this in a function
            pred_top_3 = torch.argsort(left_right_pred, dim=1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:, 0]
            ground_truth = ground_truth_left_right.argmax(dim=1)
            pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                           ground_truth, pred_top_1)

            
            if mode == "test" and calc_f1_at_k:
                # if end of video, apply_majority_voting, calculate f1@k
                # prediction_video[0].append(pred_top_1.cpu().detach().tolist()[0])
                # prediction_video[1].append(pred_top_1.cpu().detach().tolist()[1])
                subj, tsk, tk, s_frame, end_f, if_mirrored = test_idx_sample_mapping[str(idxes.item())]
                
                ifm = (if_mirrored == "mirrored") * 2 
                pred_list=pred_top_1.cpu().detach().tolist()

                prediction_video[0 + ifm][end_f]=pred_list[0]
                prediction_video[1 + ifm][end_f]=pred_list[1]
                
                # For the first N element: prediction_video[0]: left, prediction_video[1]: right
                # For the rest: prediction_video[0]: right, prediction_video[1]: left
                # becuase first not-mirrored, then mirrored
                if idxes.item() + 2*(args.temporal_length - 1) == 2*video_lengths[test_video_idx]:
                    
                    prediction_video[0][0:args.temporal_length] = prediction_video[0][args.temporal_length]
                    prediction_video[1][0:args.temporal_length] = prediction_video[1][args.temporal_length]
                    prediction_video[2][0:args.temporal_length] = prediction_video[2][args.temporal_length]
                    prediction_video[3][0:args.temporal_length] = prediction_video[3][args.temporal_length]

                    gt_video = test_ground_truth[str(test_video_idx)]

                    # calculate f1@k
                    f1_at_k(prediction_video, gt_video, segment_name=f"{subj}-{tsk}-{tk}")
                    
                    prediction_video = np.zeros((4, video_lengths[0]))
                    test_video_idx += 1

            # torchmetric
            step_f1_top1 = f1_calc_top1(left_right_pred, ground_truth)
            step_f1_top3 = f1_calc_top3(pred_top_3_final, ground_truth)

            conf_mat_top1(pred_top_1, ground_truth)
            conf_mat_top3(pred_top_3_final, ground_truth)

            # Log metrics
            if mode != "test":
                wandb.log({f"{mode}/batch/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step}, step=step)

            if mode == "test" and args.save_incorrects:
                visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, 
                                          mapping_path=opj(processed_dir, "test_idx_sample_mapping.json"))

            # if (step+1)%20==0:
            #     print("20 BATCH ÇALIŞIYOR SADECE")
            #     break

        overall_f1_top1 = f1_calc_top1.compute()
        overall_f1_top3 = f1_calc_top3.compute()

        overall_conf_mat_top1 = conf_mat_top1.compute()
        overall_conf_mat_top3 = conf_mat_top3.compute()


    if mode != "test":
        wandb.log({f"{mode}/epoch/": {'top1_F1': overall_f1_top1, 'top3_F1': overall_f1_top3}, 'epoch': epoch})

    if mode == "test" and calc_f1_at_k:
        # calculate f1@k
        f1_at_10, f1_at_25, f1_at_50 = f1_at_k.compute() 
        wandb.log({"test/f1_at_10": f1_at_10, "test/f1_at_25": f1_at_25, "test/f1_at_50": f1_at_50})


    # TODO: also check conf_mat of train. Maybe it can't overfit
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
        scores_and_conf_mats = run_one_epoch(model, test_loader, 'test', epoch=-1, calc_f1_at_k=args.f1k)
        # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = scores_and_conf_mats
        
        print(f"\tTop1: {scores_and_conf_mats[0]}, Top3: {scores_and_conf_mats[1]}")
        
        results.append(scores_and_conf_mats)
        
        if best_top1 < scores_and_conf_mats[0]:
            best_top1 = scores_and_conf_mats[0]
            best_i = i

    return results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

    assert args.gpu_id < torch.cuda.device_count(), f"gpu_id:{args.gpu_id} not found. Total gpu count: {torch.cuda.device_count()}"
    device = torch.device('cuda:'+str(args.gpu_id))
    # device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project="rt-action-segmentation-from-scene-graphs", group="after-debug", name=args.name, config=args.__dict__, 
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

    processed_dir = train_set.processed_dir

    # TODO: remove drop_last=True, does worker_init_fn work really?
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=3, worker_init_fn=seed_worker)
    
    if args.f1k:
        test_loader = DataLoader(test_set, 1, shuffle=False)
    else:
        test_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=3, worker_init_fn=seed_worker)
    
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=3, worker_init_fn=seed_worker)


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
                                     edge_dropout=args.edge_dropout,
                                     dropout=args.dropout,
                                     use_cls_token=args.use_cls_token,
                                     merged_pred=args.merged_pred).to(device=device)



    print(model)
    rnn_params = sum(p.numel() for p in model.temporal_layer.parameters() if p.requires_grad)
    gnn_params = sum(p.numel() for p in model.gnn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total: {total_params:_}")
    print(f"RNN: {rnn_params:_}")
    print(f"GNN: {gnn_params:_}")
    wandb.log({"total_params": total_params, "rnn_params": rnn_params, "gnn_params": gnn_params})


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
    
    
    
    loss_weights = get_weight_loss(train_set, args.weighted_loss)
    
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights).to(device=device)

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

    torch.cuda.empty_cache()
    wandb.finish()