# This code is like a personal attack to all programming principles, but it works

import copy
import os
import pickle
import random
from os.path import join as opj
from typing import List

import matplotlib.pyplot as plt
import matplotlib
from natsort import natsorted
import numpy as np
import torch
import cv2

from models import make_model
from dataset import get_bimanual_dataset_splits, ACTIONS, get_processing_dir, OBJECTS
from utils import arg_parser_viz_attn as arg_parser

import torch_geometric.nn as g_nn
from torch_geometric.data import Batch, Data

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.ioff()
matplotlib.use("Agg")

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)

IN_LEN = None # will be set later

# TRAIN_SUBSET = [150, 500, 1500]
# VAL_SUBSET = [50, 100, 200]
TRAIN_SUBSET = [860, 875, 890, 905]
TEST_SUBSET = [340, 355, 370, 385]

IMG_DATASET_DIR = ...

def get_imgs(sb, task, take, sfr, efr, mir, downsample):
    return None

def get_sample_batch(args):
    
    assert args.monograph, "This script is only for monograph"

    train_set, test_set, val_set = get_bimanual_dataset_splits(
        root=args.root,
        process_it_anyway=args.process_it,
        test_subject=args.test_subject,
        validation_take=args.validation_take,
        temporal_length=args.temporal_length,
        downsample=args.downsample,
        use_vf=args.use_vf,
        monograph=args.monograph,
    )

    processed_dir = get_processing_dir(
        args.root, args.temporal_length, args.downsample, args.use_vf, args.monograph
    )
    dataset_mapping_dir = opj(
        processed_dir, f"dataset_mapping_fold{args.test_subject}.pickle"
    )
    with open(dataset_mapping_dir, "rb") as f:
        dataset_mapping = pickle.load(f)

    # with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
    #     sample_dict = pickle.load(f)

    samples, labels, names, original = [], [], [], []

    # TODO load original images

    for i in TRAIN_SUBSET:
        global_idx, graph = train_set[i]

        samples.append(graph)
        labels.append(graph.y.argmax(-1))

        sb, task, take, sfr, efr, mir = dataset_mapping["train"]["sample"][global_idx]
        original.append(get_imgs(sb, task, take, sfr, efr, mir, args.downsample))
        names.append(f"{sb}{task}{take}_{sfr}_{efr}_{mir}")

    for i in TEST_SUBSET:
        global_idx, graph = test_set[i]

        samples.append(graph)
        labels.append(graph.y.argmax(-1))

        sb, task, take, sfr, efr, mir = dataset_mapping["test"]["sample"][global_idx]
        original.append(get_imgs(sb, task, take, sfr, efr, mir, args.downsample))
        names.append(f"{sb}{task}{take}_{sfr}_{efr}_{mir}")

    return samples, labels, names, original


ATTENTION = {"edge_indicies": []}  # {l1: [s1_attn, ...], ..., ln: [...]}
# li : layer i
# si_attn: attention of sample j
# CHECKME: Keeping the edge indicies might be unnecessary and redundant

def getAttentionHook(layer_name):
    def hook(model, input, output):
        assert isinstance(model, g_nn.TransformerConv)

        edge_indicies, attn = output[1]
        # attn: num_edges x num_heads
        # edge_indicies: 2 x num_edges
        # edge_indicies[:, i]: [source, target] of the i-th edge
        # sum of attentions to target is 1

        ATTENTION[layer_name].append(attn.detach().cpu())

        if layer_name == "g_layer_0":
            ATTENTION["edge_indicies"].append(edge_indicies.detach().cpu())
        
        # Just to be sure
        if layer_name == "g_layer_1":
            assert torch.all(edge_indicies.cpu() == ATTENTION["edge_indicies"][-1])

    return hook


def set_hooks(model):
    for i in range(args.num_layers):
        j=3*i
        layer_name = f"g_layer_{i}"

        getattr(model.gnn.gnn, f"module_{j}").register_forward_hook(
            getAttentionHook(f"g_layer_{i}")
        )
        ATTENTION[layer_name] = []


def rollout_graph(edges_to_preserve, attentions, discard_ratio, head_fusion):
    # attentions: List of tensor of shape [num_edges, num_heads]
    
    result = torch.ones(attentions[0].size(0))
    edges_to_preserve = torch.tensor(edges_to_preserve).reshape(-1, 1)

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # attention_heads_fused: [num_edges]

            # Drop the lowest attentions, but
            # don't drop the class token
            # flat = attention_heads_fused.view(bs, -1)
            _, indices = attention_heads_fused.topk(int(attention_heads_fused.size(-1) * discard_ratio), -1, False)

            # remove hands_indicies from indices
            indices = indices[~torch.any(indices.unsqueeze(0) == edges_to_preserve, dim=0)]
            
            attention_heads_fused[indices] = 0

            result = attention_heads_fused * result
            

    return result



def get_attn_of_an_edge(attn, edge_index, src_obj, trg_obj):
    if [src_obj, trg_obj] in edge_index:
        idx = edge_index.index([src_obj, trg_obj])
        return attn[idx]
    else:
        return 0


def viz_attn(
    save_dir,
    graph_rollout_attn,
    original_images,
    preds,
    labels,
    names,
    sample,
    edge_index,
):
    selected_frame_idx = 5
    edge_index = edge_index.t().tolist()

    # For each hand there will be one table with size of NxT. 
    # N is the number of objects available in the scene (in T frame) 
    # and T is the number of frames in the sequence.

    # First find the number of objects in the scene
    objects=sample.x.argmax(1)
    node_counts = sample.node_counts
    prev = 0
    N = []
    
    # The following two list of lists are used to keep track of the objects in the mono graph
    seperated = [] # To keep track of the objects with a type and unique name
    seperated_idx = [] # To keep track of the objects indecies in the mono graph

    for nc in node_counts: # iter over each graph
        seperated.append([])
        local_list = []
        seperated_idx.append([i for i in range(prev, prev+nc)])
        for obj in objects[prev:prev+nc].tolist(): # iter over each object in the graph
            if obj not in local_list:
                seperated[-1].append(f"{OBJECTS[obj]}_0")
                local_list.append(obj)
            else:
                seperated[-1].append(f"{OBJECTS[obj]}_{local_list.count(obj)}")
        prev += nc
    
    # N: List of unique objects in the scene
    N = list(set([item for sublist in seperated for item in sublist])) # 
    N = natsorted(N)

    # Construct the attention tables
    target_left_idx = ((objects == OBJECTS.index("LeftHand")).nonzero()[selected_frame_idx]).item()
    target_right_idx = ((objects == OBJECTS.index("RightHand")).nonzero()[selected_frame_idx]).item()

    left_attns = np.zeros((len(N), IN_LEN))
    right_attns = np.zeros((len(N), IN_LEN))
    
    # iter over each object and find the corresponding attention, and fill the table
    # with the attention values

    # Higly Very Immensely inefficient
    for i, graph_i in enumerate(seperated_idx):
        for j, obj in enumerate(graph_i):
            # obj: index of the object in the graph

            # obj_type_id: corresponding row in the attention table
            obj_type_id = N.index(seperated[i][j])

            # left hand
            left_attns[obj_type_id, i] = get_attn_of_an_edge(graph_rollout_attn, edge_index, obj, target_left_idx)

            # right hand
            right_attns[obj_type_id, i] = get_attn_of_an_edge(graph_rollout_attn, edge_index, obj, target_right_idx)


    # for obj_i, obj_j in edge_index:
    #     # left hand
    #     left_attns[N.index(f"{OBJECTS[obj_j]}_{seperated[obj_i].count(OBJECTS[obj_j])}"), obj_i] = get_attn_of_an_edge(graph_rollout_attn, edge_index, obj_i, target_left_idx)

    #     # right hand
    #     right_attns[N.index(f"{OBJECTS[obj_j]}_{seperated[obj_i].count(OBJECTS[obj_j])}"), obj_i] = get_attn_of_an_edge(graph_rollout_attn, edge_index, obj_i, target_right_idx)

    # Normalize the attention tables
    max_val = max(left_attns.max(), right_attns.max())
    left_attns = left_attns / max_val
    right_attns = right_attns / max_val

    pL, pR = preds[selected_frame_idx]
    tL, tR = labels[selected_frame_idx]
    pL, pR, tL, tR = ACTIONS[pL][:2].upper(), ACTIONS[pR][:2].upper(), ACTIONS[tL][:2].upper(), ACTIONS[tR][:2].upper()

    # Plot the attention tables
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(left_attns, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    # fig.colorbar(ax[0].imshow(left_attns, cmap="hot", interpolation="nearest", vmin=0, vmax=1), ax=ax[0])
    ax[0].set_title(f"Left F{selected_frame_idx} p({pL}) g({tL})")
    ax[0].set_yticks(np.arange(len(N)))
    ax[0].set_yticklabels(N)
    ax[0].set_xticks(np.arange(IN_LEN))
    ax[0].set_xticklabels(np.arange(IN_LEN)+1)

    # ax[1].imshow(right_attns, cmap="hot", interpolation="nearest")
    fig.colorbar(ax[1].imshow(right_attns, cmap="hot", interpolation="nearest", vmin=0, vmax=1), ax=ax[1])
    ax[1].set_title(f"Right F{selected_frame_idx} p({pR}) g({tR})")
    ax[1].set_yticks(np.arange(len(N)))
    ax[1].set_yticklabels(N)
    ax[1].set_xticks(np.arange(IN_LEN))
    ax[1].set_xticklabels(np.arange(IN_LEN)+1)

    plt.tight_layout()
    plt.savefig(opj(save_dir, f"{names}_attention_tables.png"))
    plt.close()

    # TODO images?

    


def get_edges_of_hands(sample, edge_index):
    # get index of hands in the graph
    # and find the index of edges that are connected to hands
    left_idx = OBJECTS.index("LeftHand")
    right_idx = OBJECTS.index("RightHand")

    hands = torch.logical_or(sample.x.argmax(1) == right_idx, sample.x.argmax(1) == left_idx).nonzero()
    assert hands.shape == (20, 1)
    
    edge_index = edge_index.t().tolist()
    edges_to_preserve = []
    for idx, (i, j) in enumerate(edge_index):
        if i in hands or j in hands:
            edges_to_preserve.append(idx)
    
    return edges_to_preserve


def main(args):
    global IN_LEN
    IN_LEN = args.temporal_length

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.gpu_id))

    save_dir = opj("attention_maps", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model = make_model(args).to(device)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        print("Currently no model is loaded".upper())

    set_hooks(model)

    samples, labels, names, originals = get_sample_batch(args)

    model.eval()
    # with torch.no_grad(): # This let fastpath to work, and it seems fastpath disables the hooks
    # it may not fit to the memory
    preds = []
    for sample in samples:
        sample = Batch.from_data_list([sample]).to(device)
        preds.append(model(sample.to(device)).detach().cpu().argmax(axis=-1))

    num_data = len(samples)

    # indecies of edges that are connected to hands (preserve them in the attention during discard)
    edges_to_preserve = [get_edges_of_hands(sample, edge_index) for sample, edge_index in zip(samples, ATTENTION["edge_indicies"])]

    # Each sample processed separately
    for i in range(num_data):
        attentions = [ATTENTION[f"g_layer_{j}"][i] for j in range(args.num_layers)] 
        edge_index = ATTENTION["edge_indicies"][i]

        graph_rollout_attn = rollout_graph(edges_to_preserve[i], attentions, args.discard_ratio, args.fuse_method)

        viz_attn(save_dir, graph_rollout_attn, originals[i], preds[i], labels[i], names[i], samples[i], edge_index)


if __name__ == "__main__":
    args = arg_parser()
    main(args)

