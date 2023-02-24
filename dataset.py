"""
    TODO
    - mirrored graph
    - compare with json
    - randomly check for the sequentiality
    - fix processed_file_names
    - null cases    
"""
import glob
import json
import os
from os.path import join as opj
import shutil
from typing import List, Tuple

import enlighten
import natsort
import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
from torch_geometric.loader import DataLoader

# Global variables
OBJECTS = ['bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'cup',
           'banana', 'cereals', 'sponge', 'woodenwedge', 'saw', 'harddrive', 'LeftHand',
           'RightHand']
RELATIONS = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
             'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
             'getting close', 'moving apart', 'stable'] # , 'temporal'
ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']


NONE_COUNT_IN_ACTION=0

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        # for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls']:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]: # 10 
                yield subject, task, take


class BimanualDataset(Dataset):
    def __init__(self, root: str, process_anyway: bool, temporal_lenght: int=10):
        self.temporal_lenght = temporal_lenght
        self.process_anyway = process_anyway
        print("temporal_lenght: ", temporal_lenght)
        transform, pre_transform, pre_filter=None, None, None
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return ["bimacs_rgbd_data_ground_truth/subject_1/task_1_k_cooking/take_0.json"]

    @property
    def processed_file_names(self):
        if not self.process_anyway:
            return [os.path.basename(x) for x in glob.glob(opj(self.processed_dir, 'data*.pt'))]
        
        else:
            # if os.path.isdir(self.processed_dir):
            #     shutil.rmtree(self.processed_dir)
            #     os.mkdir(self.processed_dir)        
            return "some_dummy_name.pt"

    

    def download(self):
        # Downloads Bimanual dataset to `self.raw_dir`.

        urls = ["https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_3d_objects.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_spatial_relations.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_rgbd_data_ground_truth.zip"]

        for url in urls:
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)

    # def _get_graph(self, relation_json, object_json, gt_action_json, frame) -> Data:
    def _get_graph(self, relation_json, object_json, gt_action_json, frame, subject, task, take) -> Data:
        # returns edge index and edge features
        
        # TODO: subject, task, take were just added for debugging purposes, and will be removed

        with open(relation_json) as f:
            curr_relations = json.load(f)

        with open(object_json) as f:
            curr_objects = json.load(f)

        # initialize "Data" params
        x = torch.zeros((len(curr_objects), len(OBJECTS)), dtype=torch.float)
        # y = torch.zeros((2), dtype=torch.long)
        # y = torch.zeros((2, 2), dtype=torch.long)
        y = torch.zeros((2, len(ACTIONS)), dtype=torch.long)

        edge_index = []

        num_possible_relations=len(curr_objects)*(len(curr_objects)-1)
        edge_attr = torch.zeros((int(num_possible_relations), len(RELATIONS)))

        # fill them
        for i, obj in enumerate(curr_objects):
            # one-hot encoded
            x[i, OBJECTS.index(obj["class_name"])] = 1
        
        for relation in curr_relations:
            # if relation["relation_name"]=="contact":
            oi = relation["object_index"]
            si = relation["subject_index"]


            # edge=[OBJECTS.index(curr_objects[oi]["class_name"]),
            #       OBJECTS.index(curr_objects[si]["class_name"])]
            edge=[oi, si]
            
            if edge not in edge_index:
                edge_index.append(edge)
                
            # one-hot encoded
            edge_attr[edge_index.index(edge), RELATIONS.index(relation["relation_name"])] = 1
            
    
        edge_attr=edge_attr[:len(edge_index)]
        edge_index=torch.tensor(edge_index, dtype=torch.long)
        
        left_act, right_act = self._get_actions(gt_action_json, frame)
        # y[0]=left_act
        # y[1]=right_act
        # y[0]=torch.tensor([left_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.long)
        # y[1]=torch.tensor([right_act, int(subject[-1]+task[5]+take[-1]+str(frame))],dtype=torch.long)
        y[0, left_act]=1
        y[1, right_act]=1
        

        data = Data(x=x, edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_attr, y=y)

        return data

    def _get_actions(self, gt_action_json, i) -> Tuple[int, int]:
        # Returns the action of left hand and right hand, respectively.
        global NONE_COUNT_IN_ACTION
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

        # Sometimes it gives "null" as an action label, imputes with zero (idle), and counts them
        if left_act is None:
            left_act=0
            NONE_COUNT_IN_ACTION+=1
        if right_act is None:
            right_act=0
            NONE_COUNT_IN_ACTION+=1

        return left_act, right_act

    def _get_mirrored_data(self, data) -> Data:
        # To replicate the Bimanual paper, mirror the graph i.e. switch left_hand with right_hand; right_of with left_of  
        raise NotImplementedError

    def process(self):
        derived_dir = opj(self.raw_dir, "bimacs_derived_data")
        gt_dir = opj(self.raw_dir, "bimacs_rgbd_data_ground_truth")

        idx = 0
        progressbar_manager=enlighten.get_manager()
        ticks=progressbar_manager.counter(total=540, desc="Progress:", unit="folder", color="blue")
        

        for subject, task, take in crawl_dataset():
            # make sure there is no cross-folder graph sequence
            data_list = [] 
            # data_list_mirrored = [] 

            object_dir = opj(derived_dir, subject, task, take, "3d_objects")
            object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

            relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
            relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))

            gt_action_json = opj(gt_dir, subject, task, f"{take}.json")

            assert len(relation_json_list) == len(object_json_list)

            for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):
                
                graph = self._get_graph(relation_json, object_json, gt_action_json, frame, subject, task, take)
                data_list.append(graph)
                # data_list_mirrored.append(self._get_mirrored_data(graph))

                if len(data_list) == self.temporal_lenght:
                    torch.save(data_list, opj(self.processed_dir, f'data_{idx}.pt'))
                    # torch.save(data_list, opj(self.processed_dir, f'data_{subject}_{task}_{take}_f{frame}_{idx}.pt'))
                    idx += 1
                    # torch.save(data_list_mirrored, opj(self.processed_dir, f'data_{subject}_{task}_{take}_f{frame}_{idx}.pt'))
                    # idx += 1

                    # data_list=[]
                    
                    data_list.pop(0)
                    # data_list_mirrored.pop(0)

            ticks.update()
        
        progressbar_manager.stop()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # data = torch.load(opj(self.processed_dir, f'data_{idx}.pt'))
        f=glob.glob(opj(self.processed_dir, f'data*_{idx}.pt'))
        data = torch.load(f[0])
        return data


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
        obj1, obj2=data.edge_index.t()[i]
        print("\n\n", obj1.item(), " to ", obj2.item(), ": ", end="")
        for j in torch.nonzero(data.edge_attr[i]):
            print(RELATIONS[j], end=", ")
    
    print("\n\nActions:")
    print("Left: ", ACTIONS[data.y[0,0]], ", Right: ", ACTIONS[data.y[1,0]])

    print("Graph id: ", data.y[0,1].item())



# bm = BimanualDataset(".", True, 10)

# print(f"{NONE_COUNT_IN_ACTION=}")

# loader = DataLoader(bm, batch_size=4, shuffle=True)
# batch=next(iter(loader))


# for b in batch:
#     print("#### batch ####")
#     for g in b.to_data_list():
#         print("#### gs ####")
#         print(g.y)
#         printDataPoint(g)