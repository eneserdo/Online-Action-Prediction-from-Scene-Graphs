# -*- coding: utf-8 -*-

import glob
import json
import os
import pickle
import random
import time
from datetime import datetime
from os.path import join as opj
import warnings


import numpy as np
import torch

try: 
    torch_vmap = torch.vmap
except AttributeError:
    import functorch
    torch_vmap = functorch.vmap

from torch_geometric.loader import DataLoader
# from torch_geometric.explain import Explainer, AttentionExplainer, ModelConfig

from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)

import matplotlib.pyplot as pyplot

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, get_bimanual_dataset_splits)
from utils import (F1_at_k, Metrics, VisIndividualPreds, arg_parser, calc_metrics_at_once, combine_predictions, get_loss_weights, get_minutes, metrics_to_str, plot_conf_mat, seed_worker, 
                   visualize_incorrect_preds, create_ground_truth_segments, assign_seg_labels, flatten_seq)

from tslearn.metrics import SoftDTWLossPyTorch


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


    seg_save_dir = opj(save_dir, f"test_results_{epoch}") if (mode == "test" and args.save_segments) else None
    metrics = Metrics(len(ACTIONS), args.temporal_length, device, dataset_mapping, mode, seg_save_dir, args.use_weighted_mv)

    f1_calc_top1 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)
    f1_calc_top3 = MulticlassF1Score(num_classes=len(ACTIONS), top_k=1, average="macro").to(device)  # top_k is intentionally 1

    conf_mat_top1 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)
    conf_mat_top3 = MulticlassConfusionMatrix(num_classes=len(ACTIONS), normalize="true").to(device)

    # A dictionary that will store the all predictions for all videos. This will be used to calculate f1@k
    # Used -1 as a placeholder value to check if all predictions are filled
    all_predictions_combined = {vid + m_and_hand_status: - np.ones((act.shape[0] - args.temporal_length + 1) * args.temporal_length) for vid, act in dataset_mapping[mode]["vid"].items() for m_and_hand_status in ["_L", "_R", "ML", "MR"]}


    # exp=Explainer(model.gnn, 
    #               algorithm=AttentionExplainer(), 
    #               model_config=ModelConfig(mode="multiclass_classification", task_level="node", return_type="raw"),
    #               explanation_type="model")
    

    # To ignore warnings about functorch and nan values in confusion matrix
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*There is a performance drop.*|.*NaN values found in confusion matrix.*', UserWarning)

        for idxes, data in loader:
            step += 1

            if mode == "train":
                optimizer.zero_grad()

            data = [d.to(device) for d in data]

            gt_reordered = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data], dim=0)
            # gt_reordered: [temporal_length, 2*batch_size(left, ..., left, right, ..., right) , num_classes]

            batch_size = gt_reordered.shape[1] // 2

            pred = model(data)

            pred_flattened = pred.flatten(0, 1)
            gt_flattened = gt_reordered.flatten(0, 1)
            
            label_loss = label_criterion(pred_flattened, gt_flattened)
            
            # Segmentation is currently removed
            loss = label_loss            
            
            if mode == "train":
                loss.backward()
                optimizer.step()

            # This takes 0.0024 seconds
            combine_predictions(all_predictions_combined, idxes.numpy(), pred.detach().cpu().argmax(-1), dataset_mapping[mode]["sample"])
            


            if mode != "test" and args.save_individual:
                # Visualize individiual predictions
                VIP.pick_samples(y_true=gt_reordered.cpu().argmax(2), y_pred=pred.detach().cpu(), idx=idxes, epoch=epoch, mode=mode)


            # # Calculate metrics
            # testden gelen
            pred_top_3 = torch.argsort(pred_flattened, dim=-1, descending=True)[:, :3]
            pred_top_1 = pred_top_3[:,0]
            ground_truth = gt_flattened.argmax(dim=-1)
            pred_top_3_final = torch.where(torch_vmap(lambda t1, t2: torch.isin(t1, t2))(ground_truth, pred_top_3),
                                        ground_truth, pred_top_1)
            

            # torchmetric
            # TODO: Create f1 for each position in the sequence separately
            step_f1_top1 = f1_calc_top1(pred_flattened, ground_truth) # NOTE: Lookup if input can have more than 2 channels
            step_f1_top3 = f1_calc_top3(pred_top_3_final, ground_truth)

            conf_mat_top1(pred_top_1, ground_truth)
            conf_mat_top3(pred_top_3_final, ground_truth)

            step_f1_top1_, step_f1_top3_ = metrics.update(pred.detach(), gt_reordered, idxes)
            assert np.allclose([step_f1_top1_, step_f1_top3_], step_f1_top1, step_f1_top3)

            if mode != "test":
                wandb.log({f"{mode}/batch/": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step})

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

    
    
    f1_at_10, f1_at_25, f1_at_50, f1_top1_once, f1_top3_once, _ = calc_metrics_at_once(all_predictions_combined, 
                                                                dataset_mapping[mode]["vid"], 
                                                                num_classes=len(ACTIONS), 
                                                                temporal_length=args.temporal_length,
                                                                use_weighted_mv=args.use_weighted_mv,
                                                                save_dir=seg_save_dir)
    

    metric_dict = metrics.compute()

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
    
    # FIXME: adapt for floats
    assert metric_dict["f1@10"] == return_dict["f1@10"]
    assert metric_dict["f1@25"] == return_dict["f1@25"]
    assert metric_dict["f1@50"] == return_dict["f1@50"]
    assert metric_dict["f1_top1"] == return_dict["f1_top1"]
    assert metric_dict["f1_top3"] == return_dict["f1_top3"]
    assert metric_dict["cm_top1"] == return_dict["cm_top1"]
    assert metric_dict["cm_top3"] == return_dict["cm_top3"]
    assert metric_dict["f1_top1_once"] == return_dict["f1_top1_once"]

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

    #### asserts about args:

    if args.task != "onetoone":
        raise NotImplementedError("Currently only onetoone task is supported")

    if args.perfect_seg:
        raise KeyError("perfect_seg is deprecated. Use --task seg instead.")
    
    if args.task in ["labelonly", "both", "onetoone"] and args.label_loss_weight == 0:
        raise ValueError(f"task {args.task} requires label_loss_weight to be non-zero")
    
    if not args.disable_seg:
        raise NotImplementedError("Segmentation may be broken, please check it")

    assert args.merged_pred in ["none", "late", "attention"], f"merged_pred must be one of [none, late, attention]. Given: {args.merged_pred}"

    # Make sure it is reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(0)

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

    train_loader = DataLoader(train_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)
    
    test_loader = DataLoader(test_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    val_loader = DataLoader(val_set, args.batch_size, num_workers=3, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    training_len = len(train_set)
    val_len = len(val_set)

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
                                     task=args.task,
                                     merged_pred=args.merged_pred,
                                     norm_layer=args.norm_layer,
                                     DISABLE_SEGMENTATION=args.disable_seg,
                                     norm_first=args.norm_first,
                                     pos_enc=args.pos_enc,
                                     temporal_length=args.temporal_length).to(device=device)

    print(model)
    rnn_params = sum(p.numel() for p in model.temporal_layer.parameters() if p.requires_grad)
    gnn_params = sum(p.numel() for p in model.gnn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable params: {total_params:_}")
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


    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = getattr(torch.optim, args.optimizer_type)(model.parameters(), lr=args.learning_rate,
                                                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size,
                                                gamma=args.scheduler_gamma, verbose=True)


    # loss weights can be used to balance the classes using different strategies
    loss_weights = get_loss_weights(args.weighted_loss, train_set.distribution)


    label_criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.smoothing).to(device=device)
    
    # if args.seg_loss == "BCELoss":
    #     segmentation_criterion = torch.nn.BCELoss(weight=loss_weights).to(device=device)
    # elif args.seg_loss == "sDTW":
    #     # TODO: try different distance measures... cross-entropy??
    #     segmentation_criterion = SoftDTWLossPyTorch(gamma=args.sdtw_gamma)

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
        
    if args.save_individual:
        VIP = VisIndividualPreds(20, training_len, val_len, save_dir)

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
        if epoch % args.saving_freq == 0 or epoch == start + args.max_epochs - 1:
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

        print(f'### {epoch:03d}: Time {get_minutes(stop - start)} m ### Train: ', metrics_to_str(train_metrics), ' ### Val: ', metrics_to_str(val_metrics))


    # save the individual predictions
    if args.save_individual:
        VIP.save()

    # TEST
    print("Starts test...")
    # f_top1_test, f_top3_test, conf_mat_top1_test, conf_mat_top3_test = run_one_epoch(model, test_loader, 'test', epoch=-1)
    test_metrics = test_for_all_saved_weights(model, test_loader)


    plt = plot_conf_mat(test_metrics["cm_top1"], f"TEST_conf_mat_top1_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top1': wandb.Image(plt)}})
    plt3 = plot_conf_mat(test_metrics["cm_top3"], f"TEST_conf_mat_top3_e{epoch:0>4}", do_return_plot=True)
    wandb.log({'test/confusion_matrix/': {'top3': wandb.Image(plt3)}})
    

    print(f'### [F1] TEST: ', metrics_to_str(test_metrics))
    wandb.log({"test_top1": round(test_metrics["f1_top1"], 4), "test_top3": round(test_metrics["f1_top3"], 4), "f1_top1_once": round(test_metrics["f1_top1_once"], 4), "f1@10": round(test_metrics["f1@10"], 4), "f1@25": round(test_metrics["f1@25"], 4), "f1@50": round(test_metrics["f1@50"], 4)})

    plt.close()
    plt3.close()
    torch.cuda.empty_cache()
    wandb.finish()