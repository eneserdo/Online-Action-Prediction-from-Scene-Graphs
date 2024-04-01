# Gets all arguments and start subprocesses using those arguments
# collect all outputs and calculate mean and std for each metric
# choose best one and log the mean and std to wandb

from functools import partial
import os
from os.path import join as opj
import subprocess
import time
import numpy as np
import pickle
import wandb

from utils import arg_parser_cv

METRICS_OF_INTEREST = ["f1_top1_once", "f1_top3_once", "f1@10", "f1@25", "f1@50"]
COMPARISON_METRIC = "f1_top1_once"


def main(args, search_params):

    args = vars(args)
    args.update(search_params)

    # Get all arguments as a string except the name
    args_str = ""
    for key, value in args.__dict__.items():
        if key == "name":
            continue

        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                args_str += f" --{key}"
        else:
            args_str += f" --{key} {value}"

    for i in range(1, 1 + args.num_folds):
        print(
            " #######################\n",
            "####### FOLD",
            i,
            "########\n",
            "#######################\n",
        )

        saved_dir = f"saved_{args.name}_fold{i}"
        cv_file = opj(saved_dir, f"cross_validation_fold{i}.pickle")

        if os.path.exists(cv_file):
            print(f"Fold {i} already exists. Skipping...")
            continue

        subject_arg = f" --test_subject {i}"
        name_arg = f"{args.name}_fold{i}"

        subprocess.call(
            "python main.py " + args_str + subject_arg + f" --name {name_arg}",
            shell=True,
        )
        time.sleep(5)

    all_outputs = []  # list of list of dictionaries

    for i in range(1, 1 + args.num_folds):
        saved_dir = f"saved_{args.name}_fold{i}"
        cv_file = opj(saved_dir, f"cross_validation_fold{i}.pickle")

        with open(cv_file, "rb") as f:
            data = pickle.load(f)

        all_outputs.append(data)

    num_weights = len(data)

    processed_outputs = []
    final_metrics = []

    for _ in range(num_weights):
        processed_outputs.append({key: [] for key in METRICS_OF_INTEREST})
        final_metrics.append({key: {} for key in METRICS_OF_INTEREST})

    for e in range(num_weights):
        for k in METRICS_OF_INTEREST:
            for i in range(args.num_folds):
                processed_outputs[e][k].append(all_outputs[i][e][k])

            arr = np.array(processed_outputs[e][k])
            final_metrics[e][k]["mean"] = np.mean(arr)
            final_metrics[e][k]["std"] = np.std(arr)

    if args.num_folds > 1:
        cv_n = f"CV{args.num_folds}"
        args.name = cv_n + "_" + args.name


        if args.sweep:
            project_name = "sweep-rt-graph"
        else:
            project_name = "rt-action-segmentation-from-scene-graphs"

        wandb.init(
            project=project_name,
            name=args.name,
            config=args.__dict__,
            mode="disabled" if args.disable_wandb else "online",
            tags=[cv_n],
        )

        best_e = 0
        best_score = 0
        for e in range(num_weights):
            if best_score < final_metrics[e][COMPARISON_METRIC]["mean"]:
                best_score = final_metrics[e][COMPARISON_METRIC]["mean"]
                best_e = e

        metrics = final_metrics[best_e]
        for key in metrics.keys():
            mean = metrics[key]["mean"]
            std = metrics[key]["std"]

            # I preferred them to appear under the same column in wandb
            # even though they have different number of folds
            key = "CV_" + key
            print(f"{key} mean: {mean}, std: {std}")
            wandb.log({f"{key}_mean": mean, f"{key}_std": std})

        wandb.finish()


if __name__ == "__main__":

    args = arg_parser_cv()

    if args.sweep:
        sweep_config = {"method": "random",
                        "metric": {"name": "f1_top1_once_mean", "goal": "maximize"}}

        # Search space
        search_parameters = {
            # optimization parameters
            "learning_rate": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            "weight_decay": {"values": [0.0, 1e-4, 1e-5, 1e-6]},
            "momentum": {"values": [0.9, 0.99]},

            # graph parameters
            "gnn_type": {"values": ["TransformerConv", "GATv2Conv"]},
            "hidden_channels": {"values": [32, 64, 96, 128]},
            "out_channels": {"values": [32, 64, 96, 128]},
            "num_heads1": {"values": [2, 4, 6, 8]},
            "num_heads2": {"values": [2, 4, 6, 8]},
            "edge_dropout": {"values": [0, 0.05, 0.1, 0.15, 0.2]},
            "dropout": {"values": [0, 0.05, 0.1, 0.15, 0.2]},
            "merged_pred": {"values": ["none", "late", "attention"]},
            "norm_layer": {"values": ["BatchNorm", "InstanceNorm", "LayerNorm", "GraphNorm", "PairNorm", "none"]},
            
            # transformer parameters
            "num_heads_tr": {"values": [2, 4, 6, 8]},
            "dim_feedforward": {"values": [64, 96, 128, 192]},
            "dropout_tr": {"values": [0, 0.1, 0.15, 0.2, 0.25]},
            "num_of_temp_layers": {"values": [1, 2, 3]},

            # input parameters
            "use_pos": {"values": [True, False]},
            "use_embedding": {"values": [True, False]},

            # loss parameters
            "smoothing": {"values": [0.0, 0.05, 0.15]},
        }

        sweep_config['parameters'] = search_parameters

        sweep_id = wandb.sweep(sweep_config, project="sweep-rt-graph")

        main_with_args = partial(main, args=args)
        
        wandb.agent(sweep_id, main_with_args, count=5)

    else:
        main(args, {})