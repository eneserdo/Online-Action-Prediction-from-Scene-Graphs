import argparse
from os.path import join as opj
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from dataset import ACTIONS

plt.ioff()

def arg_parser():

    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default='.', help="Root directory of dataset.")
    ap.add_argument("-p", "--process_it", action="store_true", help="...")
    ap.add_argument("-contact", "--contact_only", action="store_true", help="...")
    ap.add_argument("-bs", "--batch_size", type=int, default=256, help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")

    # Model
    ap.add_argument("-g", "--gnn_type", type=str, required=True, choices=["NNConv", "GATConv", "TransformerConv", "GATv2Conv"], help="...")
    ap.add_argument("-hc", "--hidden_channels", type=int, default=64, help="...")
    ap.add_argument("-oc", "--out_channels", type=int, default=64, help="...")
    ap.add_argument("-hs", "--lstm_hidden_size", type=int, default=128, help="...")
    ap.add_argument("-nh", "--num_heads", type=int, default=2, help="...")
    ap.add_argument("--saving_freq", type=int, default=5, help="...")
    ap.add_argument("-nbn", "--no_bn", action="store_true", help="...")
    ap.add_argument("--dropout", type=float, default=0.0, action="store_true", help="...")

    # Optimizer
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="...")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="...")
    ap.add_argument("--scheduler_step_size", type=int, default=10, help="...")
    ap.add_argument("--scheduler_gamma", type=float, default=0.5, help="...")

    # Training
    ap.add_argument("-me", "--max_epochs", type=int, default=50, help="Max number of epochs")
    ap.add_argument("-r", "--restore", type=int, help="...")
    ap.add_argument("-rn", "--restored_name", type=str, help="...")

    # Logging
    ap.add_argument("-n", "--name", required=True, type=str, help="Name of experiment for logging into Wandb")
    ap.add_argument("--disable_wandb",action="store_true", help="Disable wandb")

    return ap.parse_args()


def arg_parser_inference():

    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--folder_path", required=True, type=str, help="...")    
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")

    ap.add_argument("-gt", "--ground_truth_path", help="...")

    # Model
    ap.add_argument("-ch", "--conv_hidden", type=int, default=64, help="...")
    ap.add_argument("-co", "--conv_out", type=int, default=64, help="...")
    ap.add_argument("-hs", "--hidden_size", type=int, default=128, help="...")
    ap.add_argument("-r", "--restore", type=int, required=True, help="...")

    return ap.parse_args()


def conf_mat_from_left_right(truth, pred, labels):
    #pred_left, pred_right = pred[:batch_size], pred[batch_size:]
    #truth_left, truth_right = truth[::2], truth[1::2]


    #res = confusion_matrix(truth_left.cpu().detach().numpy(), pred_left.cpu().detach().numpy(), labels=[range(0,len(ACTIONS)]) + \
    #  confusion_matrix(truth_right.cpu().detach().numpy(), pred_right.cpu().detach().numpy(), labels=[range(0,len(ACTIONS)])

    #pred = torch.cat([pred[:batch_size], pred[batch_size:]])
    truth = torch.cat([truth[::2], truth[1::2]])

    res = confusion_matrix(truth.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=labels)

    return res


def classification_scores(conf_mat):
    scores = {'recall': {}, 'precision': {}, 'F1': {}}

    TP = np.diagonal(conf_mat)
    TP_p_FN = np.sum(conf_mat, axis=1)  # total class true occurrences
    TP_p_FP = np.sum(conf_mat, axis=0)  # total class predicted occurrences

    mask1 = (TP_p_FN != 0)
    # recall = TP / (TP + FN)
    scores['recall']['all'] = TP[mask1] / TP_p_FN[mask1]

    mask2 = (TP_p_FP != 0)
    # precision = TP / (TP + FP)
    scores['precision']['all'] = TP[mask2] / TP_p_FP[mask2]

    #  mask??

    # F1 = 2 * TP / (2 * TP + FN + FP)
    scores['F1']['all'] = 2 * TP / (TP_p_FP + TP_p_FN)

    # Micro average
    scores['recall']['avg']={}
    scores['precision']['avg']={}
    scores['F1']['avg']={}

    sum_labels = np.sum(TP_p_FN)
    accuracy = np.sum(TP) / sum_labels
    scores['accuracy'] = accuracy
    scores['recall']['avg']['micro'] = accuracy
    scores['precision']['avg']['micro'] = accuracy
    scores['F1']['avg']['micro'] = accuracy

    # Macro average
    n_labels = len(TP)
    scores['recall']['avg']['macro'] = np.sum(scores['recall']['all']) / n_labels
    scores['precision']['avg']['macro'] = np.sum(scores['precision']['all']) / n_labels
    scores['F1']['avg']['macro'] = np.sum(scores['F1']['all']) / n_labels

    # Weighted average
    scores['recall']['avg']['weighted'] = np.sum(scores['recall']['all'] * TP_p_FN) / sum_labels
    scores['precision']['avg']['weighted'] = np.sum(scores['precision']['all'] * TP_p_FN) / sum_labels
    scores['F1']['avg']['weighted'] = np.sum(scores['F1']['all'] * TP_p_FN) / sum_labels

    # Normalized confusion matrix
    scores['normalized confusion matrix'] = np.divide(conf_mat, conf_mat.sum(axis=1,keepdims=True))

    return scores


def get_minutes(elapsed) -> str:
    return f"{round((elapsed)//60)}.{round((elapsed)%60)}"

def plot_conf_mat(mat, name, do_save=False, do_return_plot=False):
    mat_long=torch.round(mat*100).long()

    plt.figure(figsize=(14,12))
    sns.heatmap(mat_long, square=True, annot=True, cmap='Blues', cbar=False, xticklabels=ACTIONS, yticklabels=ACTIONS, vmin=0, vmax=100)
    plt.yticks(rotation=0)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    if do_save:
        raise NotImplementedError("conf_mat diye bir klasor yok artik")
        plt.savefig(opj("conf_mat", f'{name}.png'))
    if do_return_plot:
        return plt
    else:
        plt.close()
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
