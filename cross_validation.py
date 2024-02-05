
# get all arguments and start subprocesses using those arguments

import pickle
import sys
import subprocess
import time
from os.path import join as opj

import numpy as np


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python cross_validation.py <args>")
        sys.exit(1)

    name_index = args.index("--name") if "--name" in args else args.index("-n") if "-n" in args else None

    name = args[name_index + 1]

    args = " ".join(args)

    for i in range(1, 7):
        subject_arg = f" --test_subject {i}" 
        subprocess.call("python main.py " + args + subject_arg, shell=True)
        time.sleep(5)

    metrics = {"f1_top1_once": [], "f1_top3_once": [], "f1@10": [], "f1@25": [], "f1@50": []}

    for i in range(1, 7):
        cv_file = opj(name, f"cross_validation_fold{i}.pickle")

        with open(cv_file, "rb") as f:
            data = pickle.load(f)
            
            for key in metrics.keys():
                metrics[key].append(data[key])


    with open(opj(name, "results.txt"), "w") as f:
        for key in metrics.keys():
            arr = np.array(metrics[key])
            mean = np.mean(arr)
            std = np.std(arr)
            f.write(f"{key} mean: {mean}, std: {std}\n")
            print(f"{key} mean: {mean}, std: {std}")

        

if __name__ == "__main__":
    main()