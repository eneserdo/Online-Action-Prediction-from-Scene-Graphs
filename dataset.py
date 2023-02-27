"""
    TODO
    - mirrored graph
    - compare with json
    - randomly check for the sequentiality
    - fix processed_file_names
    - remove data.verify 
"""
import glob
import json
import os
import shutil
from os.path import join as opj
from typing import List, Tuple
import json

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


MISSING_HAND_COUNT=0
NULL_COUNT=0
TOTAL_FRAME_COUNT=0
EMPTY_COUNT=0
NO_EDGE_COUNT=0

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        # for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls']:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]: # 10 
                yield subject, task, take


class BimanualDataset(Dataset):
    def __init__(self, *, root: str, is_processed: bool, split: str, test_subject: int, validation_take: int , temporal_length: int):
        self.temporal_length = temporal_length
        self.is_processed = is_processed
        self.test_subject = test_subject
        self.validation_take = validation_take
        self.split = split
        self.config = None
        transform, pre_transform, pre_filter=None, None, None
        self.deleted_once=False
        
        assert split in ["val", "train", "test"], "split can only be val, train or test"
        if is_processed:
            self._load_config()
            assert self.config["test_subject"]==test_subject
            assert self.config["validation_take"]==validation_take

        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return ["bimacs_rgbd_data_ground_truth/subject_1/task_1_k_cooking/take_0.json"]

    @property
    def processed_file_names(self):
        print("processed_file_names called")
        
        if self.is_processed:
            # addr = [os.path.basename(x) for x in glob.glob(opj(self.processed_dir, 'data*.pt'))]
            # addr = [opj(self.split, a) for a in addr]
            addr = [opj("train", 'data_0.pt'), opj("test", 'data_0.pt'), opj("val", 'data_0.pt')]
            return addr
        else:
            if os.path.isdir(self.processed_dir):
                if not self.deleted_once:
                    print("Deleting processed_dir...")
                    shutil.rmtree(self.processed_dir)
                    os.mkdir(self.processed_dir)
                    os.mkdir(opj(self.processed_dir, "train"))
                    os.mkdir(opj(self.processed_dir, "test"))
                    os.mkdir(opj(self.processed_dir, "val"))
                    self.deleted_once=True
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
        
        with open(relation_json) as f:
            curr_relations = json.load(f)

        with open(object_json) as f:
            curr_objects = json.load(f)

        # initialize "Data" params
        x = torch.zeros((len(curr_objects), len(OBJECTS)), dtype=torch.float)
        # y = torch.zeros((2, 2), dtype=torch.long)
        y = torch.zeros((2, len(ACTIONS)), dtype=torch.float)

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

        # subject, task, take were just added for debugging purposes, and will be removed
        # y[0]=torch.tensor([left_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.long)
        # y[1]=torch.tensor([right_act, int(subject[-1]+task[5]+take[-1]+str(frame))],dtype=torch.long)

        if left_act is not None:
            y[0, left_act]=1

        if right_act is not None:
            y[1, right_act]=1
        

        data = Data(x=x, edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_attr, y=y)

        return data

    def _get_actions(self, gt_action_json: os.PathLike, i: int) -> Tuple[int, int]:
        # Returns the action of left hand and right hand, respectively.

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

    def _get_mirrored_data(self, data: Data) -> Data:
        # To replicate the Bimanual paper, mirror the graph i.e. switch left_hand with right_hand; right_of with left_of  
        raise NotImplementedError

    def _check_validity(self, data: Data) -> bool:
        # Checks: both hands exists, action != null, has edge, not empty
        
        data.validate(raise_on_error=True) 
        
        global MISSING_HAND_COUNT, NULL_COUNT, EMPTY_COUNT, NO_EDGE_COUNT

        Lhand, Rhand=False, False
        # null
        if torch.sum(data.y[0])==0 or torch.sum(data.y[1])==0:
            NULL_COUNT+=1
            return False

        # not empty
        if len(data.x)==0:
            EMPTY_COUNT+=1
            return False
        
        # both hands
        for row in data.x:
            if row[OBJECTS.index("LeftHand")]==1:
                Lhand=True
            
            if row[OBJECTS.index("RightHand")]==1:
                Rhand=True
                
        if not (Rhand and Lhand):
            MISSING_HAND_COUNT+=1
            return False


        # has edge
        if len(data.edge_index)==0:
            NO_EDGE_COUNT+=1
            return False
        
        return True

    def _save_config(self, config_info: dict) -> None:
        with open(opj(self.processed_dir, "config_info.json"), "w") as f:
            json.dump(config_info, f)
        
        self.split_len=self.config[self.split]
            
    def _load_config(self) -> None:
        with open(opj("processed", "config_info.json"), "r") as f:
            self.config = json.load(f)
        
        self.split_len=self.config[self.split]

    def process(self):
        global TOTAL_FRAME_COUNT
        derived_dir = opj(self.raw_dir, "bimacs_derived_data")
        gt_dir = opj(self.raw_dir, "bimacs_rgbd_data_ground_truth")

        curr_subj=0

        train_sample_idx=0
        val_sample_idx=0
        test_sample_idx=0

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
                
                TOTAL_FRAME_COUNT+=1
                graph = self._get_graph(relation_json, object_json, gt_action_json, frame, subject, task, take)

                # if the graph is not approptiate, resets the sequence and continues
                if not self._check_validity(graph):
                    data_list=[]
                    # print("Skipped: ", subject, task, take, "f", frame)
                    continue    

                data_list.append(graph)
                # data_list_mirrored.append(self._get_mirrored_data(graph))

                if len(data_list) == self.temporal_length:
                    
                    if int(subject[-1])==self.test_subject:
                        torch.save(data_list, opj(self.processed_dir, "test", f'data_{test_sample_idx}.pt'))
                        test_sample_idx+=1
                    
                    elif int(take[-1])==self.validation_take:
                        torch.save(data_list, opj(self.processed_dir, "val", f'data_{val_sample_idx}.pt'))
                        val_sample_idx+=1

                    else:
                        torch.save(data_list, opj(self.processed_dir, "train", f'data_{train_sample_idx}.pt'))
                        train_sample_idx+=1

                    # torch.save(data_list, opj(self.processed_dir, sub_dir, f'data_{idx}.pt'))
                    # idx += 1
                    # torch.save(data_list_mirrored, opj(self.processed_dir, f'data_{subject}_{task}_{take}_f{frame}_{idx}.pt'))
                    # idx += 1
                    
                    data_list.pop(0)
                    # data_list_mirrored.pop(0)

            ticks.update()

            # if curr_subj != subject:
            #     curr_subj=subject
            #     print(f"\n\n####### {subject}\n{TOTAL_FRAME_COUNT=}, {EMPTY_COUNT=}, {NO_EDGE_COUNT=}, {MISSING_HAND_COUNT=}, {NULL_COUNT=}")
        
        progressbar_manager.stop()
        print(f"\n\n#######\n{TOTAL_FRAME_COUNT=}, {EMPTY_COUNT=}, {NO_EDGE_COUNT=}, {MISSING_HAND_COUNT=}, {NULL_COUNT=}")
        
        self.config = {"test": test_sample_idx, 
                       "train": train_sample_idx, 
                       "val": val_sample_idx, 
                       "test_subject": self.test_subject, 
                       "validation_take": self.validation_take}
        
        print(self.config)
        
        self._save_config(self.config)
        self.is_processed=True

    def len(self):
        # return len(self.processed_file_names)
        return self.split_len

    def get(self, idx):
        # data = torch.load(opj(self.processed_dir, f'data_{idx}.pt'))
        # f=glob.glob(opj(self.processed_dir, f'data*_{idx}.pt'))
        # data = torch.load(f[0])

        data = torch.load(opj(self.processed_dir, self.split, f'data_{idx}.pt'))
        return data



def get_bimanual_dataset_splits(*, root: str, is_processed: bool, test_subject: int, validation_take: int , temporal_length: int) -> Tuple[Dataset, Dataset, Dataset]:
    # Wrapper function to prepare dataset splits (train, test, validation)
    
    train_set=BimanualDataset(root=root, is_processed=is_processed, split="train", test_subject=test_subject, validation_take=validation_take, temporal_length=temporal_length)
    test_set=BimanualDataset(root=root, is_processed=True, split="test", test_subject=test_subject, validation_take=validation_take, temporal_length=temporal_length)
    val_set=BimanualDataset(root=root, is_processed=True, split="val", test_subject=test_subject, validation_take=validation_take, temporal_length=temporal_length)
    
    return train_set, test_set, val_set


# bm = BimanualDataset(root=".", split="train",  test_subject=1, validation_take=7, is_processed=False, temporal_length=10)

# # train_pseudo=bm[500:]
# # test_pseudo=bm[:500]

# loader = DataLoader(bm, batch_size=4, shuffle=True)
# batch=next(iter(loader))

# for b in batch:
#     print("#### batch ####")
#     for g in b.to_data_list():
#         print("#### gs ####")
#         print(g.y)