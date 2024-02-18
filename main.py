import glob
import os
import pickle
import random
import time
from os.path import join as opj
import warnings

import numpy as np
import torch


from torch_geometric.loader import DataLoader
# from torch_geometric.explain import Explainer, AttentionExplainer, ModelConfig

import matplotlib.pyplot as pyplot

import models
import wandb
from dataset import (ACTIONS, OBJECTS, RELATIONS, _get_max_length, get_bimanual_dataset_splits, get_processing_dir)
from utils import (F1_at_k, Metrics, VisIndividualPreds, almost_equal, arg_parser, calc_metrics_at_once, combine_predictions, get_loss_weights, get_minutes, metrics_to_str, plot_conf_mat, seed_worker, 
                   visualize_incorrect_preds, create_ground_truth_segments, assign_seg_labels, flatten_seq, wandb_logging)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # To fix the following error:
# OSError: [Errno 24] Too many open files
# Alternatively ulimit can be increased
torch.multiprocessing.set_sharing_strategy('file_system')

# # An attempt to fix the following error:
# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:10.0" after 2627 requests (2627 known processed) with 8 events remaining.
pyplot.ioff()

# To check for NaNs in the gradients
# torch.autograd.set_detect_anomaly(True)


def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch: int):
    global step

    if mode in ["test", "val"]:
        model.eval()
    elif mode == "train":
        model.train()


    seg_save_dir = opj(save_dir, f"test_results_{epoch}") if (mode == "test" and args.save_segments) else None
    metrics = Metrics(len(ACTIONS), args.temporal_length, device, dataset_mapping, mode, seg_save_dir, args.weighted_mv)

    # exp=Explainer(model.gnn, 
    #               algorithm=AttentionExplainer(), 
    #               model_config=ModelConfig(mode="multiclass_classification", task_level="node", return_type="raw"),
    #               explanation_type="model")
    

    for unpackable in loader:
        if args.temporal_length == -1:
            vid_lens, idxes, data = unpackable
            # Create a padding masks, since the temporal length is not fixed
            # Am I genius or what? :D
            padding_mask = torch.arange(0, MAX_VID_LEN).expand(len(idxes), MAX_VID_LEN)
            padding_mask = padding_mask >= vid_lens.unsqueeze(1)
            padding_mask = padding_mask.to(device)
            # mask: [MAX_VID_LEN, batch_size] (even though data is [temporal_length, batch_size, feature_size])
        else:
            idxes, data = unpackable
            padding_mask = None
            padding_mask_reshaped = None

        step += 1

        if mode == "train":
            optimizer.zero_grad()

        data = [d.to(device) for d in data]

        gt_reordered = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data], dim=0)
        # gt_reordered: [temporal_length, 2*batch_size(left, ..., left, right, ..., right) , num_classes]

        batch_size = gt_reordered.shape[1] // 2

        pred = model(data, padding_mask)

        
        pred_flattened = pred.flatten(0, 1)
        gt_flattened = gt_reordered.flatten(0, 1)
        
        if padding_mask is not None:
            # Arange the padding mask to match the pred_flattened and gt_flattened
            padding_mask_reshaped=padding_mask.T
            padding_mask_reshaped= torch.cat([padding_mask_reshaped, padding_mask_reshaped], dim=1)
            padding_mask_reshaped=padding_mask_reshaped.flatten()
            pred_flattened = pred_flattened[~padding_mask_reshaped]
            gt_flattened = gt_flattened[~padding_mask_reshaped]
        
        label_loss = label_criterion(pred_flattened, gt_flattened)
        
        # Segmentation is currently removed
        loss = label_loss
        
        if mode == "train":
            loss.backward()
            optimizer.step()


        if mode != "test" and args.save_individual:
            # Visualize individiual predictions
            raise NotImplementedError("save_individual may be broken due to global idx, please check it")
            VIP.pick_samples(y_true=gt_reordered.cpu().argmax(2), y_pred=pred.detach().cpu(), idx=idxes, epoch=epoch, mode=mode)


        step_f1_top1, step_f1_top3 = metrics.update(pred.detach(), gt_reordered, idxes, padding_mask_reshaped)


        if mode != "test":
            wandb.log({f"{mode}/batch": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step})

        if mode == "test" and args.save_incorrects:
            raise NotImplementedError("save_incorrects may be broken, please check it")
            visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, 
                                        test_idx_sample_mapping=dataset_mapping["test"]["sample"])
        # if (step+1)%20==0:
        #     print("20 BATCH ÇALIŞIYOR SADECE")
        #     break

    # End of for loop
    
    metric_dict = metrics.compute()


    return metric_dict


def test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of):
    # Log everything for cross-validation 

    cv_file = opj(save_dir, f"cross_validation_fold{args.test_subject}.pickle")
    cv_list = []

    comparison_metric=choose_best_in_terms_of
    
    weight_list=sorted(glob.glob(opj(save_dir, "*.pt")))

    if len(weight_list) == 0:
        return None
    
    best_metric = 0    
    results = []
    best_i = -1
    for i, w_name in enumerate(weight_list):
        print(f"TESTING {w_name}:")
        checkpoint = torch.load(w_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        returned_dict = run_one_epoch(model, test_loader, 'test', epoch=int((w_name[-7:-3])))
        cv_list.append(returned_dict)
        
        print(f'### TEST: ', metrics_to_str(returned_dict))

        results.append(returned_dict)        
        if best_metric < returned_dict[comparison_metric]:
            best_metric = returned_dict[comparison_metric]
            best_i = i

    with open(cv_file, "wb") as f:
        pickle.dump(cv_list, f)

    print(f'### BEST: ', metrics_to_str(results[best_i]))
    return results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

    #### asserts about args:

    if args.temporal_length ==-1:
        assert args.norm_layer == "LayerNorm", "When temporal_length is -1, norm_layer must be LayerNorm, otherwise it may break the norm stats"
        assert args.temporal_type == "tr", "others are not implemented yet"

        
    assert args.merged_pred in ["none", "late", "attention"], f"merged_pred must be one of [none, late, attention]. Given: {args.merged_pred}"

    # Make sure it is reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)

    # assert args.gpu_id < torch.cuda.device_count(), f"gpu_id:{args.gpu_id} not found. Total gpu count: {torch.cuda.device_count()}"
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    wandb.init(project="rt-action-segmentation-from-scene-graphs", name=args.name, config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online")


    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, 
                                                               process_it_anyway=args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               downsample=args.downsample)

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
                                     num_heads1=args.num_heads1,
                                     num_heads2=args.num_heads2,
                                     filter_it=args.filter_idle_hold,
                                     use_hand_pooling=not args.use_global_pooling,
                                     edge_dropout=args.edge_dropout,
                                     dropout=args.dropout,
                                     dropout_tr=args.dropout_tr,
                                     merged_pred=args.merged_pred,
                                     norm_layer=args.norm_layer,
                                     norm_first=args.norm_first,
                                     pos_enc=args.pos_enc,
                                     temporal_length=args.temporal_length,
                                     num_of_temp_layers=args.num_of_temp_layers,
                                     dim_feedforward=args.dim_feedforward,
                                     num_heads_tr=args.num_heads_tr).to(device=device)

    # print(model)
    rnn_params = sum(p.numel() for p in model.temporal_layer.parameters() if p.requires_grad)
    gnn_params = sum(p.numel() for p in model.gnn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Total trainable params: {total_params:_}")
    # print(f"RNN: {rnn_params:_}")
    # print(f"GNN: {gnn_params:_}")
    wandb.log({"total_params": total_params, "rnn_params": rnn_params, "gnn_params": gnn_params})


    wandb.watch(model, log='all')
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch*", step_metric="epoch")
    wandb.define_metric("val/epoch*", step_metric="epoch")
    wandb.define_metric("test/epoch*", step_metric="epoch")
    # wandb.define_metric("test*", step_metric="epoch")
    # wandb.define_metric("val/epoch/*", step_metric="epoch", summary="max")
    wandb.define_metric("val/confusion_matrix*", step_metric="epoch")
    wandb.define_metric("test/confusion_matrix*", step_metric="epoch")
    wandb.define_metric("batch")
    wandb.define_metric("train/batch*", step_metric="batch")
    wandb.define_metric("val/batch*", step_metric="batch")
    # wandb.define_metric("test_top1")
    # wandb.define_metric("test_top3")



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


    save_dir = f"saved_{args.name}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    # Load saved models if required
    if args.restore is not None:
        assert args.restored_name is not None, "restored_name must be provided if restore is not None"
        checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Model is restored successfully from epoch {args.restore}!")
        
        # FIXME: It overrides the scheduler. Is it desired?
        print("Optimizer and scheduler are overwritten with the restored ones")

    
    # dataset_mapping contains mapping for each sample to its location in the videos. 
    # Also, a mapping from video to ground truth action labels
    processed_dir = get_processing_dir(args.root, args.temporal_length, args.downsample)
    dataset_mapping_dir = opj(processed_dir, f"dataset_mapping_fold{args.test_subject}.pickle")
    with open(dataset_mapping_dir, "rb") as f:
        dataset_mapping = pickle.load(f)
        
    if args.save_individual:
        VIP = VisIndividualPreds(20, training_len, val_len, save_dir)

    if args.temporal_length == -1:
        MAX_VID_LEN = _get_max_length(args.root, args.downsample)

    # STARTS TRAINING AND VALIDATION
    step = 0
    print("Starts training...")
    start = 1 if args.restore is None else args.restore
    epoch = start
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        train_metrics = run_one_epoch(model, train_loader, 'train', epoch)
        wandb_logging(wandb, train_metrics, epoch, "train", whole_mode=args.temporal_length == -1)
        
        val_metrics = run_one_epoch(model, val_loader, 'val', epoch)
        wandb_logging(wandb, val_metrics, epoch, "val", whole_mode=args.temporal_length == -1)

        # Save model
        if epoch % args.saving_freq == 0 or epoch == start + args.max_epochs - 1:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                        opj(save_dir, f"model_epoch{epoch:0>4}.pt"))


        scheduler.step()
        stop = time.perf_counter()

        print(f'### {epoch:03d}: Time {get_minutes(stop - start)} m ### Train: ', metrics_to_str(train_metrics), ' ### Val: ', metrics_to_str(val_metrics))


    # save the individual predictions
    if args.save_individual:
        VIP.save()

    # TEST
    print("Starts test...")

    choose_best_in_terms_of = "f1_top1_once" if args.temporal_length != -1 else "f1_top1"
    test_metrics = test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of=choose_best_in_terms_of)
    
    if test_metrics is None:
        print("[WARN] No weights found to test")
    else:
        wandb_logging(wandb, test_metrics, epoch, "test", whole_mode=args.temporal_length == -1)
    
    wandb.finish()
    
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            time.sleep(5)
        else:
            raise e