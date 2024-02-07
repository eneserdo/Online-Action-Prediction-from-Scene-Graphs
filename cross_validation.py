# get all arguments and start subprocesses using those arguments
# collect all outputs and calculate mean and std for each metric
# choose best one and log the mean and std to wandb

from os.path import join as opj
import subprocess
import time
import numpy as np
import pickle
import wandb

from utils import arg_parser

METRICS_OF_INTEREST = ["f1_top1_once", "f1_top3_once", "f1@10", "f1@25", "f1@50"]
COMPARISON_METRIC = "f1_top1_once"

def main():
    args = arg_parser()

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


    for i in range(1, 7):
        
        subject_arg = f" --test_subject {i}"
        name_arg = f"{args.name}_fold{i}"

        # print("python main.py " + args_str + subject_arg + f" --name {name_arg}\n\n\n")
        print(" #######################\n",
              "####### FOLD", i, "########\n",
              "#######################\n")
        subprocess.call("python main.py " + args_str + subject_arg + f" --name {name_arg}", shell=True)
        time.sleep(5)


    all_outputs = [] # list of list of dictionaries


    for i in range(1, 7):
        saved_dir = f"saved_{args.name}_fold{i}"
        cv_file = opj(saved_dir, f"cross_validation_fold{i}.pickle")

        with open(cv_file, "rb") as f:
            data = pickle.load(f)
            
        all_outputs.append(data)

    num_weights = len(data)

    processed_outputs = []
    final_metrics = []

    for _ in range(num_weights):
        processed_outputs.append({ key: [] for key in METRICS_OF_INTEREST })
        final_metrics.append({ key: {} for key in METRICS_OF_INTEREST })

    for e in range(num_weights):
        for k in METRICS_OF_INTEREST:
            for i in range(6):
                processed_outputs[e][k].append(all_outputs[i][e][k])
            
            arr = np.array(processed_outputs[e][k])
            final_metrics[e][k]["mean"] = np.mean(arr)
            final_metrics[e][k]["std"] = np.std(arr)
    
    args.name = "CV_" + args.name

    wandb.init(project="rt-action-segmentation-from-scene-graphs", 
               name=args.name, 
               config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online",
               tags=["CV"])
    

    best_e=0
    best_score=0
    for e in range(num_weights):
        if best_score < final_metrics[e][COMPARISON_METRIC]["mean"]:
            best_score = final_metrics[e][COMPARISON_METRIC]["mean"]
            best_e = e

    metrics= final_metrics[best_e]
    for key in metrics.keys():
        mean = metrics[key]["mean"]
        std = metrics[key]["std"]

        key = "CV_" + key
        print(f"{key} mean: {mean}, std: {std}")
        wandb.log({f"{key}_mean": mean, f"{key}_std": std})

    wandb.finish()


if __name__ == "__main__":
    main()