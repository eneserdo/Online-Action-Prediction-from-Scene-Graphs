import argparse

import torch
from torch_geometric.data import Data

from dataset import ACTIONS, OBJECTS, RELATIONS


def arg_parser():

    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-bs", "--batch_size", type=int, default=16, help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, help="...")

    # Model
    # ap.add_argument("-nef", "--num_edge_feat", type=int, default=32, help="...")
    # ap.add_argument("-ci", "--conv_in", type=int, default=32, help="...")
    ap.add_argument("-ch", "--conv_hidden", type=int, default=32, help="...")
    ap.add_argument("-co", "--conv_out", type=int, default=32, help="...")
    ap.add_argument("-hs", "--hidden_size", type=int, default=64, help="...")

    return ap.parse_args()


# Simple tests
def printDataPoint(data: Data):

    print("Summary:")
    print(f"{data.num_nodes=}")
    print(f"{data.num_edges=}")
    print(f"{data.num_node_features=}")
    print(f"{data.is_directed()=}")

    print("validate: ", data.validate(raise_on_error=True))

    print("\nObjects:")
    for i in torch.argmax(data.x, dim=1):
        print(OBJECTS[i])

    assert data.edge_index.shape[1] == data.edge_attr.shape[0]

    print("\nEdges:", end="")
    for i in range(len(data.edge_index.t())):
        obj1, obj2 = data.edge_index.t()[i]
        print("\n\n", obj1.item(), " to ", obj2.item(), ": ", end="")

        for j in torch.nonzero(data.edge_attr[i]):
            print(RELATIONS[j], end=", ")

    print("\n\nActions:")
    print("Left: ", ACTIONS[torch.argmax(data.y[0], dim=1)], ", Right: ", ACTIONS[torch.argmax(data.y[0], dim=1)])

    # print("Graph id: ", data.y[0,1].item())
