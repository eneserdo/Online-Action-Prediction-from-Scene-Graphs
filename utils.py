import argparse
import glob
import json
import os
from os.path import join as opj
import random
import shutil

import functorch
import matplotlib.pyplot as plt
import natsort
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
    ap.add_argument("--fix_fps", action="store_true", help="...")
    ap.add_argument("-bs", "--batch_size", type=int, default=256, help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")

    # Model
    ap.add_argument("-g", "--gnn_type", type=str, required=True, choices=["NNConv", "GATConv", "TransformerConv", "GATv2Conv"], help="...")
    ap.add_argument("--temporal_type", type=str, default="lstm", choices=["lstm", "bi", "tr"], help="...")
    ap.add_argument("-hc", "--hidden_channels", type=int, default=64, help="...")
    ap.add_argument("-oc", "--out_channels", type=int, default=64, help="...")
    ap.add_argument("-hs", "--lstm_hidden_size", type=int, default=128, help="...")
    ap.add_argument("-nh", "--num_heads", type=int, default=2, help="...")
    ap.add_argument("--saving_freq", type=int, default=5, help="...")
    ap.add_argument("-nbn", "--no_bn", action="store_true", help="...")
    ap.add_argument("--use_global_pooling", action="store_true", help="...")
    ap.add_argument("--edge_dropout", type=float, default=0.0, help="...")
    ap.add_argument("--dropout", type=float, default=0.0, help="...")
    ap.add_argument("--use_cls_token", action="store_true", help="Use class token in transformer")

    # Optimizer
    ap.add_argument("-o", "--optimizer_type", type=str, required=True,
                    choices=["Adam", "AdamW", "NAdam", "RMSprop", "SGD"],
                    help="Type of optimizer to be used  from torch.optim", default='Adam')
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="...")
    ap.add_argument("-mn", "--momentum", type=float, default=0.9, help="...")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="...")
    ap.add_argument("--scheduler_step_size", type=int, default=10, help="...")
    ap.add_argument("--scheduler_gamma", type=float, default=0.5, help="...")
    ap.add_argument("--weighted_loss", type=str, default="none", choices=["none", "inv", "sqrt_inv" "effective"], help="Use class frequencies as weight for loss function")

    # Training
    ap.add_argument("-me", "--max_epochs", type=int, default=20, help="Max number of epochs")
    ap.add_argument("-r", "--restore", type=int, help="...")
    ap.add_argument("-rn", "--restored_name", type=str, help="...")
    ap.add_argument("-fih", "--filter_idle_hold", action="store_true", help="...")

    # Logging
    ap.add_argument("-n", "--name", required=True, type=str, help="Name of experiment for logging into Wandb")
    ap.add_argument("--disable_wandb",action="store_true", help="Disable wandb")

    # Others
    ap.add_argument("--gpu_id", type=int, default=0, help="...")


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

def predicted_labels(pred_scores, corect_labels, k: float):
    # pred_scores -> (n_samples, n_labels)
    # corect_labels -> (n_samples,)
    # k -> IoU expected to consider a prediction correct
    pred_top_3 = torch.argsort(pred_scores, dim=1, descending=True)[:, :3]
    pred_top_1 = pred_top_3[:, 0]
    pred_top_3_final = torch.where(functorch.vmap(lambda t1, t2: torch.isin(t1, t2))(corect_labels, pred_top_3),
                                   corect_labels, pred_top_1)
    # F1 @ k metric -> commonly k = 0.1, 0.25 and 0.5
    pred_at_k = torch.where(pred_scores[corect_labels] >= k, corect_labels, pred_top_1)

    return pred_top_1, pred_top_3_final, pred_at_k

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
    scores['recall']['avg'] = {}
    scores['precision']['avg'] = {}
    scores['F1']['avg'] = {}

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
    scores['recall']['avg']['weighted'] = np.sum(scores['recall']['all'] * TP_p_FN[mask1]) / sum_labels
    scores['precision']['avg']['weighted'] = np.sum(scores['precision']['all'] * TP_p_FN[mask2]) / sum_labels
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



def get_filters_for_imbalance(dataBatchList):
    """ 2 out of 3 samples for the actions idle and hold were discarded """ 

    batch_size=dataBatchList[0].y.shape[0]
    left_filter=torch.full((len(dataBatchList), batch_size//2), True) # shape = [sequence_length, batch_size]
    right_filter=torch.full((len(dataBatchList), batch_size//2), True) # shape = [sequence_length, batch_size]
    

    for i in range(batch_size//2):
        count_idle_left=0
        count_idle_right=0
        count_hold_left=0
        count_hold_right=0

        for t, dataBatch in enumerate(dataBatchList):
            count_idle_left = (count_idle_left + torch.argmax(dataBatch[i].y[0])==ACTIONS.index("idle")) % 3
            count_idle_right = (count_idle_right + torch.argmax(dataBatch[i].y[1])==ACTIONS.index("idle")) % 3
            count_hold_left = (count_hold_left + torch.argmax(dataBatch[i].y[0])==ACTIONS.index("hold")) % 3
            count_hold_right = (count_hold_right + torch.argmax(dataBatch[i].y[1])==ACTIONS.index("hold")) % 3

            left_filter[t, i] = not (count_idle_left in [1, 2])
            right_filter[t, i] = not (count_idle_right in [1, 2])
            
            left_filter[t, i] = not (count_hold_left in [1, 2])
            right_filter[t, i] = not (count_hold_right in [1, 2])
            
    return left_filter, right_filter


def align_embeddings_to_right(x, mask):
    """
        Ref 1: https://discuss.pytorch.org/t/reorder-non-zero-values-in-a-2d-matrix/112314
        Ref 2: https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    """

    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values, to fix it I used "stable=True"
    # Still suspicious
    index = torch.sort(~mask.repeat(1,1,emd_size).to(x.device) * 1, dim=0, descending=True, stable=True)[1]
    y = x.gather(0, index)

    # d_mask = torch.sum(mask, dim=0, keepdims=True) > torch.arange(x.shape[0]).reshape(-1,1)
    # y = torch.zeros_like(x)
    # y[d_mask] = x[mask]

    return y


def debug_fih(dataBatchList, b):
    
    for i in range(10):
        
        # hold
        if torch.argmax(dataBatchList[i].y[b]).item()==5:
            print("h ", end="")
        # idle
        elif torch.argmax(dataBatchList[i].y[b]).item()==0:
            print("i ", end="")
        else:
            print(torch.argmax(dataBatchList[i].y[b]).item(), end=" ")

    print("")    


def print_non_uniform_ys(dataBatchList):
    bs=dataBatchList[0].y.shape[0]
    for b in range(bs):
        act=torch.argmax(dataBatchList[0].y[b]).item()
        for i in range(10):
            if act != torch.argmax(dataBatchList[i].y[b]).item():

                print("b id: ", b)
                debug_fih(dataBatchList, b)
                break


def debug_align(device="cpu"):
    mask = torch.full((4, 3), True)
    mask[0, 0]=False
    mask[2, 2]=False
    mask[3, 0]=False
    mask[3, 1]=False
    
    # print("mask:\n", mask)
    
    x = torch.arange(12).reshape(4,3,1)
    x = x.repeat(1, 1, 2)
    x[~mask]=0

    x = x.to(device)
    mask = mask.to(device)
    
    # t=4, bs=3, emb_size=2
    
    # print("x:\n", x)
    # print("x[:, :, 0]:\n", x[:, :, 0])

    # y = align_embeddings_to_right(x, mask)
    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values
    index = torch.sort(~mask.repeat(1,1,emd_size).to(x.device) * 1, dim=0, descending=True)[1]
    y = x.gather(0, index)


    # print("y[:, :, 0]:\n", y[:, :, 0])

    return x, y, index, mask



def get_relations(base_dir, subject, task, take, i):
    """ Returns a set for relations in the frame """
    
    # derived_dir=base_dir+"bimacs_derived_data/"
    derived_dir=opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path=natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))


    with open(relations_path[i]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[i]) as f:
        _3d = json.load(f)

    
    # log=""
    # text=""
    readable_relations=[]

    for relation in relations:
        oi=relation["object_index"]
        si=relation["subject_index"]

        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "
        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "

        readable_relations.append(_3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"])

        # if relation["relation_name"]=="contact":
            
            # text=text + f"({oi})"+object_mapping[oi] + " - " + f"({si})"+ object_mapping[si] + "\n"
            # if oi < si:
            #     text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"
            
            

    # print("FRAME:\n", log, "\n")
    return readable_relations


def visualize_incorrect_preds(preds, labels, idxes, mapping_path, graph_dataset_dir="raw", rgb_dataset_dir="/home/stannis/enes/bimanual/bimacs_rgbd_data"):
    
    assert os.path.exists(rgb_dataset_dir), "RGB dataset directory does not exist"
    assert os.path.exists(graph_dataset_dir), "Graph dataset directory does not exist"

    save_dir_name = "incorrect_predictions"
    # if os.path.exists(save_dir_name):
        # print("Removing existing", save_dir_name)
        # shutil.rmtree(save_dir_name)
    os.makedirs(save_dir_name, exist_ok=True)

    with open(mapping_path, "r") as f:
        test_idx_sample_mapping = json.load(f)

    incorrect_mask = preds != labels
    preds, labels, idxes = preds[incorrect_mask], labels[incorrect_mask], idxes[incorrect_mask]

    num_of_samples = 10
    random_samples=np.random.randint(0, len(preds), num_of_samples)
    preds, labels, idxes = preds[random_samples], labels[random_samples], idxes[random_samples]

    
    for idx, p, gt in zip(idxes, preds, labels):
        subject, task, take, start_frame, end_frame, is_mirrored = test_idx_sample_mapping[idx]
        sample_save_dir = opj(save_dir_name, f"{ACTIONS[p]}_{ACTIONS[gt]}_{idx}")
        os.mkdir(sample_save_dir)

        # Copy frames
        for frame in range(start_frame, end_frame+1):    
            chunk_no = frame//100
            frame_path = opj(rgb_dataset_dir, subject, task, take, "rgb", f"chunk_{chunk_no}", f"frame_{frame}.png")
            shutil.copy(frame_path, sample_save_dir)

        # Open a text file, and save relations
        with open(opj(sample_save_dir, "relations.txt"), "w") as f:
            prev_relations = []
            for frame in range(start_frame, end_frame+1):
                f.write(f"Frame {frame}:\n\n")

                curr_relations = get_relations(graph_dataset_dir, subject, task, take, frame)
                
                diff_relations=list(set(curr_relations) - set(prev_relations))

                for relation in diff_relations:
                    f.write(relation + "; ")
                f.write("\n")

                prev_relations = curr_relations
