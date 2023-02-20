#%%
import glob
import json
import os
from os.path import join as opj
from typing import List, Tuple
import warnings
import enlighten
import natsort
import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip

OBJECTS = ['bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'cup',
           'banana', 'cereals', 'sponge', 'woodenwedge', 'saw', 'harddrive', 'LeftHand',
           'RightHand']
RELATIONS = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
             'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
             'getting close', 'moving apart', 'stable', 'temporal']
ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']


print("change subject")
def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 2)]:
        # for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
        #              'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
        #              'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
        for task in ['task_1_k_cooking', 'task_6_w_hard_drive', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(3)]: # 10 
                yield subject, task, take


class BimanualDataset(Dataset):
    def __init__(self, root=".", temporal_lenght=10):
        self.temporal_lenght = temporal_lenght
        print("temporal_lenght: ", temporal_lenght)
        transform, pre_transform, pre_filter=None, None, None
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return ["bimacs_rgbd_data_ground_truth/subject_1/task_1_k_cooking/take_0.json"]

    @property
    def processed_file_names(self):
        # TODO
        return [f'data_{i}.pt' for i in range(1)]
    

    def download(self):
        # Download to `self.raw_dir`.
        urls = ["https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_3d_objects.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_spatial_relations.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_rgbd_data_ground_truth.zip"]

        for url in urls:
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)

    def _get_graph(self, relation_json, object_json, gt_action_json, frame) -> Data:
        # returns edge index and edge features

        # read jsons
        with open(relation_json) as f:
            curr_relations = json.load(f)

        with open(object_json) as f:
            curr_objects = json.load(f)

        # initialize "Data" params
        x = torch.zeros((len(curr_objects), len(OBJECTS)), dtype=torch.float)
        y = torch.zeros((2, len(ACTIONS)), dtype=torch.float)
        # FIXME: Muhtemelen yanlış
        # edge_index = torch.zeros((len(curr_relations), 2), dtype=torch.long) # TRANSPOSED LATER ON!
        # edge_attr = torch.zeros((len(curr_relations), len(RELATIONS)))

        # Önce varmış gibi oluştur sonra boşları slicing ile atarsın (alternatifi hiç oluşturmayıp tek tek concatlamak eklemek?)
        num_possible_relations=len(curr_objects)*(len(curr_objects)-1)
        edge_index = [] # TRANSPOSED LATER ON!
        edge_attr = torch.zeros((int(num_possible_relations), len(RELATIONS)))

        # fill them
        for i, obj in enumerate(curr_objects):
            # one-hot encoded
            x[i, OBJECTS.index(obj["class_name"])] = 1

        # for i, relation in enumerate(curr_relations):
        
        for relation in curr_relations:
            # if relation["relation_name"]=="contact":
            oi = relation["object_index"]
            si = relation["subject_index"]

            # if oi < si:
            # text=text + objects[oi]["class_name"]+ " - " + objects[si]["class_name"] +"\n"
            # edge_index[i, 0] = OBJECTS.index(curr_objects[oi]["class_name"])
            # edge_index[i, 1] = OBJECTS.index(curr_objects[si]["class_name"])
            edge=[OBJECTS.index(curr_objects[oi]["class_name"]),
                  OBJECTS.index(curr_objects[si]["class_name"])]
            
            if edge not in edge_index:
                edge_index.append(edge)
                
            # one-hot encoded
            edge_attr[edge_index.index(edge), RELATIONS.index(relation["relation_name"])] = 1
            

        edge_attr=edge_attr[:len(edge_index)]
        edge_index=torch.tensor(edge_index)
        
        left_act, right_act = self._get_actions(gt_action_json, frame)
        y[0, left_act]=1
        y[1, right_act]=1

        data = Data(x=x, edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_attr, y=y)
        return data

    def _get_actions(self, gt_action_json, i) -> Tuple[int, int]:

        with open(gt_action_json) as f:
            actions_json = json.load(f)

        # right
        for t in range(0, len(actions_json["right_hand"]) - 1, 2):

            begin = actions_json["right_hand"][t]
            right_act = actions_json["right_hand"][t + 1]
            end = actions_json["right_hand"][t + 2]

            if begin <= i < end:
                break

        if i == actions_json["right_hand"][-1]:
            right_act = actions_json["right_hand"][-2]

        # left
        for t in range(0, len(actions_json["left_hand"]) - 1, 2):

            begin = actions_json["left_hand"][t]
            left_act = actions_json["left_hand"][t + 1]
            end = actions_json["left_hand"][t + 2]

            if begin <= i < end:
                break

        if i == actions_json["left_hand"][-1]:
            left_act = actions_json["left_hand"][-2]

        return left_act, right_act

    def process(self):
        derived_dir = opj(self.raw_dir, "bimacs_derived_data")
        gt_dir = opj(self.raw_dir, "bimacs_rgbd_data_ground_truth")

        idx = 0
        progressbar_manager=enlighten.get_manager()
        ticks=progressbar_manager.counter(total=540, desc="Progress:", unit="folder", color="blue")

        # for raw_path in self.raw_paths:
        for subject, task, take in crawl_dataset():
            # Read data from `raw_path`.

            object_dir = opj(derived_dir, subject, task, take, "3d_objects")
            object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

            relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
            relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))

            gt_action_json = opj(gt_dir, subject, task, f"{take}.json")

            assert len(relation_json_list) == len(object_json_list)
            data_list = []

            for frame, (relation_json, object_json) in enumerate(zip(relation_json_list[:-self.temporal_lenght], object_json_list[:-self.temporal_lenght])):

                data_list.append(self._get_graph(relation_json, object_json, gt_action_json, frame))

                if len(data_list) % self.temporal_lenght == 0:
                    # torch.save(data_list, opj(self.processed_dir, f'data_{idx}.pt'))
                    torch.save(data_list, opj(self.processed_dir, f'data_{subject}_{task}_{take}_f{frame}_{idx}.pt'))
                    idx += 1

            ticks.update()
        
        progressbar_manager.stop()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # data = torch.load(opj(self.processed_dir, f'data_{idx}.pt'))
        f=glob.glob(opj(self.processed_dir, f'data*_{idx}.pt'))
        data = torch.load(f)
        return data


# Simple tests
def printDataPoint(data: Data):

    print("Summary:")
    print(f"{data.num_nodes=}")
    print(f"{data.num_edges=}")
    print(f"{data.num_node_features=}")
    print(f"{data.is_directed()=}")

    print("validate: ", data.validate(raise_on_error=True))
    
    print("Objects:")
    for i in torch.argmax(data.x, dim=1):
        print(OBJECTS[i])

    assert len(data.edge_index) == len(data.edge_attr)

    print("Edges:")
    for i in range(len(data.edge_index)):
        obj1, obj2=data.edge_index[i]
        print(obj1, " to ", obj2, ":\n")
        for j in torch.nonzero(data.edge_attr):
            print(RELATIONS[j], end=", ")
        print("")
    
    print("Actions:")
    act=torch.argmax(data.y, dim=1)
    print("Left: ", ACTIONS[act[0]], "Right: ", ACTIONS[act[1]])

    
bm = BimanualDataset(".")

# %%

# print(bm.num_classes)
# print(bm.num_node_features)
from torch_geometric.loader import DataLoader

loader = DataLoader(bm, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)
    print(batch.num_graphs)
    break

# %%

# data.num_nodes
# >>> 3

# data.num_edges
# >>> 4

# data.num_node_features
# >>> 1

# data.has_isolated_nodes()
# >>> False

# data.has_self_loops()
# >>> False

# data.is_directed()