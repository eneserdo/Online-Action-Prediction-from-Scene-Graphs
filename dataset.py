"""
    TODO
    - mirrored graph
    - remove data.verify and unnecessary asserts
    - bimanual paper sample count: 442216 vs our: 421610 

    - !!! prepating dataset and getting Dataset instances should be separate
"""
import glob
import json
import os
import shutil
from os.path import join as opj
from typing import List, Tuple
import json
import gdown

import enlighten
import natsort
import numpy as np
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



def crawl_dataset_FULL():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take

def crawl_dataset_DEBUG():
    for subject in ['subject_1', 'subject_2']: # (1, 7)
        # for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls']:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_1', 'take_7']: # 10
                yield subject, task, take

DEBUG_MODE = False

if DEBUG_MODE:
    print("Debug mode is on\n"*5)
    crawl_dataset = crawl_dataset_DEBUG
else:
    crawl_dataset = crawl_dataset_FULL



def _get_actions_list(gt_action_json: os.PathLike, act_len: int) -> Tuple[int, int]:
    # Returns the action of left hand and right hand, respectively.

    with open(gt_action_json) as f:
        actions_json = json.load(f)


    actions_list=np.zeros((act_len, 2), dtype=np.int32)

    # right
    for t in range(0, len(actions_json["right_hand"]) - 1, 2):

        begin = actions_json["right_hand"][t]
        right_act = actions_json["right_hand"][t + 1]
        end = actions_json["right_hand"][t + 2]

        right_act = -1 if right_act == None else right_act

        actions_list[begin:end, 1]=right_act

    # left
    for t in range(0, len(actions_json["left_hand"]) - 1, 2):

        begin = actions_json["left_hand"][t]
        left_act = actions_json["left_hand"][t + 1]
        end = actions_json["left_hand"][t + 2]

        left_act = -1 if left_act == None else left_act
        actions_list[begin:end, 0]=left_act
    
    return actions_list


# def _get_graph(self, relation_json, object_json, gt_action_json, frame) -> Data:
def _get_graph(relation_json, object_json, gt_actions, frame, subject, task, take) -> Tuple[Data, Data]:
    # returns edge index and edge features
    
    with open(relation_json) as f:
        curr_relations = json.load(f)

    with open(object_json) as f:
        curr_objects = json.load(f)

    # initialize "Data" params
    x = torch.zeros((len(curr_objects), len(OBJECTS)), dtype=torch.float)
    x_mirrored = torch.zeros((len(curr_objects), len(OBJECTS)), dtype=torch.float)
    
    y = torch.zeros((2, len(ACTIONS)), dtype=torch.float)
    y_mirrored = torch.zeros((2, len(ACTIONS)), dtype=torch.float)

    # To debug add embed some info
    # y = torch.zeros((2, len(ACTIONS)+1), dtype=torch.float)
    # y_mirrored = torch.zeros((2, len(ACTIONS)+1), dtype=torch.float)

    edge_index = []
    edge_index_mirrored = []

    num_possible_relations=len(curr_objects)*(len(curr_objects)-1)

    edge_attr = torch.zeros((int(num_possible_relations), len(RELATIONS)))
    edge_attr_mirrored = torch.zeros((int(num_possible_relations), len(RELATIONS)))

    ### Fill them

    for i, obj in enumerate(curr_objects):
        x[i, OBJECTS.index(obj["class_name"])] = 1

        # # # Mirrorred # # #
        if obj["class_name"]=='LeftHand':
            x_mirrored[i, OBJECTS.index('RightHand')] = 1

        elif obj["class_name"]=='RightHand':
            x_mirrored[i, OBJECTS.index('LeftHand')] = 1

        else:
            x_mirrored[i, OBJECTS.index(obj["class_name"])] = 1
        # # # # # # # # # # # #
    
    for relation in curr_relations:

        # if self.contact_only and (relation["relation_name"] != "contact"):
        #     # Skips the other relations if desired
        #     continue

        oi = relation["object_index"]
        si = relation["subject_index"]

        edge=[oi, si]

        if edge not in edge_index:
            edge_index.append(edge)

        edge_attr[edge_index.index(edge), RELATIONS.index(relation["relation_name"])] = 1
        
        # # # Mirrorred # # #
        if edge not in edge_index_mirrored:
            edge_index_mirrored.append(edge)
        
        if relation["relation_name"]=='right of':
            edge_attr_mirrored[edge_index_mirrored.index(edge), RELATIONS.index('left of')] = 1

        elif relation["relation_name"]=='left of':
            edge_attr_mirrored[edge_index_mirrored.index(edge), RELATIONS.index('right of')] = 1

        else:
            edge_attr_mirrored[edge_index_mirrored.index(edge), RELATIONS.index(relation["relation_name"])] = 1
        # # # # # # # # # # # #
        

    edge_attr=edge_attr[:len(edge_index)]
    edge_index=torch.tensor(edge_index, dtype=torch.long)

    # # # Mirrorred # # #
    edge_attr_mirrored=edge_attr_mirrored[:len(edge_index_mirrored)]
    edge_index_mirrored=torch.tensor(edge_index_mirrored, dtype=torch.long)
    # # # # # # # # # # # #

    left_act, right_act = gt_actions

    # subject, task, take were just added for debugging purposes, and will be removed
    # y[0]=torch.tensor([left_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.float)
    # y[1]=torch.tensor([right_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.float)

    graph_str=subject[-1]+task[5]+take[-1]+str(frame)

    if left_act != -1:
        y[0, left_act]=1
        y_mirrored[1, left_act]=1
        
        # To debug
        # y[0, -1]=int("1"+graph_str)
        # y_mirrored[1, -1]=int("2"+graph_str)


    if right_act != -1:
        y[1, right_act] = 1
        y_mirrored[0, right_act] = 1

        # To debug
        # y[1, -1] = int("1"+graph_str)
        # y_mirrored[0, -1] = int("2"+graph_str)
    

    data = Data(x=x, edge_index=edge_index.t().contiguous(),
                edge_attr=edge_attr, y=y)

    data_mirrored = Data(x=x_mirrored, edge_index=edge_index_mirrored.t().contiguous(),
                            edge_attr=edge_attr_mirrored, y=y_mirrored)

    return data, data_mirrored


def _check_validity(data: Data) -> str:
    # Checks: both hands exists, action != null, has edge, not empty 
    
    global MISSING_HAND_COUNT, NULL_COUNT, EMPTY_COUNT, NO_EDGE_COUNT

    Lhand, Rhand=False, False
    # null
    if torch.sum(data.y[0])==0 or torch.sum(data.y[1])==0:
        NULL_COUNT+=1
        raise Exception("Null action")
        return "null"

    # not empty
    if len(data.x)==0:
        EMPTY_COUNT+=1
        return "empty"
    
    # both hands
    for row in data.x:
        if row[OBJECTS.index("LeftHand")]==1:
            Lhand=True
        
        if row[OBJECTS.index("RightHand")]==1:
            Rhand=True
            
    if not (Rhand and Lhand):
        MISSING_HAND_COUNT+=1
        return "missing hand"

    # has edge
    if len(data.edge_index)==0:
        NO_EDGE_COUNT+=1
        return "no edge"
    
    return "valid"


def _impute_with_previous(graph: Data, previous_graph: Data, validity: str) -> Data:
    # For now, just returns the previous graph
    # If required, impute using the data of previous graph
    return previous_graph.clone()
    

class BimanualDataset(Dataset):
    def __init__(self, root: str, 
                       is_processed: bool, 
                       split: str, 
                       test_subject: int, 
                       validation_take: int , 
                       temporal_length: int, 
                       contact_only: bool, 
                       fix_fps: bool=False,
                       fix_null: bool=False):
        
        if fix_fps:
            assert temporal_length%2==0, "When fix fps enabled, temporal_length must be even"
        
        self.temporal_length = temporal_length
        self.is_processed = is_processed
        self.test_subject = test_subject
        self.validation_take = validation_take
        self.split = split
        self.config = None
        transform, pre_transform, pre_filter=None, None, None
        self.deleted_once=False
        self.fix_fps=fix_fps
        self.fix_null=fix_null

        self.root = root
        self.contact_only=contact_only
        self.distribution = torch.zeros((len(ACTIONS)))

        assert split in ["val", "train", "test"], "split can only be val, train or test"
        if is_processed:
            self._load_config()
            assert self.config["test_subject"]==test_subject
            assert self.config["validation_take"]==validation_take
            assert self.config["temporal_length"]==temporal_length


        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self) -> str:
        return opj(self.root, 'processed'+str(self.temporal_length))

    @property
    def raw_file_names(self):
        return ["bimacs_rgbd_data_ground_truth/subject_1/task_1_k_cooking/take_0.json"]

    @property
    def processed_file_names(self):        
        if self.is_processed:
            # addr = [os.path.basename(x) for x in glob.glob(opj(self.processed_dir, 'data*.pt'))]
            # addr = [opj(self.split, a) for a in addr]
            addr = [opj("train", 'data_0.pt'), opj("test", 'data_0.pt'), opj("val", 'data_0.pt')]
            return addr
        else:
            if os.path.isdir(self.processed_dir):
                print()
                if not self.deleted_once:
                    print("Deleting processed_dir...")
                    shutil.rmtree(self.processed_dir)
                    os.mkdir(self.processed_dir)
                    self.deleted_once=True
            return "some_dummy_name.pt"

    def _fix_null(self):
        # Somehow left hand in the video of subject_6-task_4-take_1 is all null. This is to fix it

        def read_check_write(sub, task, tk_json, hand, new_vals):
            # An internal function to read, check and write the json file
            # Read
            the_json_path=opj(self.raw_dir, "bimacs_rgbd_data_ground_truth", sub, task, tk_json)
            with open(the_json_path, "r") as f:
                the_dict = json.load(f)

            # Check
            old_time_steps = the_dict[hand][::2]
            new_time_steps = new_vals[::2]
            
            assert old_time_steps == new_time_steps, f"{old_time_steps=}, {new_time_steps=}"
            
            none_found=False
            act_labels = the_dict[hand][1::2]
            for i in act_labels:
                if i==None:
                    none_found=True
                    break
            
            if not none_found:
                print("No null found: ", the_json_path)
                return  # no need to fix            

            # Write back
            the_dict[hand] = new_vals
            with open(the_json_path, "w") as f:
                json.dump(the_dict, f)
            
            print("Fixed: ", the_json_path)


        read_check_write("subject_6", "task_4_k_wiping", "take_1.json", "left_hand", 
                         [0, 0, 91, 1, 114, 3, 129, 5, 150, 4, 165, 2, 189, 1, 209, 5, 357])
        read_check_write("subject_1", "task_9_w_sawing", "take_2.json", "right_hand", 
                         [0, 0, 14, 1, 32, 3, 42, 9, 97, 9, 133])
        read_check_write("subject_1", "task_7_w_free_hard_drive", "take_7.json", "right_hand", 
                         [0, 0, 13, 1, 28, 3, 36, 2, 39, 1, 44, 3, 56, 11, 135, 2, 138, 1, 142, 3, 154, 0, 181, 1, 207, 11, 285, 2, 290, 1, 292, 3, 300, 4, 309, 4, 331, 2, 333, 1, 337, 4, 350, 2, 358, 0, 381])
        read_check_write("subject_3", "task_2_k_cooking_with_bowls", "take_1.json", "right_hand", 
                         [0, 0, 61, 1, 85, 5, 194, 1, 199, 1, 206, 3, 241, 6, 339, 4, 366, 2, 371, 1, 376, 5, 475, 2, 521, 0, 542])

        print("# Replaced null actions by manually detected labels (This is directly done on the raw data!)")

    def download(self):
        # Downloads Bimanual dataset to `self.raw_dir`.

        urls = ["https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_3d_objects.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_spatial_relations.zip",
                "https://bimanual-actions.humanoids.kit.edu/download/bimacs_rgbd_data_ground_truth.zip"]

        for url in urls:
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
        
        # Features https://drive.google.com/drive/folders/1yfwItIoQrAnbnk5GTjbbfN8Ls8Ybl_hr

        gdown.download(id="1FceipuZMUjp1ES2THqlAmQgPdE18-flk", output=opj(self.raw_dir, "BimanualActions.zip"))
        extract_zip(opj(self.raw_dir, "BimanualActions.zip"), self.raw_dir)
        self._fix_null()
        
    def _save_config(self, config_info: dict) -> None:
        with open(opj(self.processed_dir, "config_info.json"), "w") as f:
            json.dump(config_info, f)
        
        self.split_len=self.config[self.split]
        
    def _load_config(self) -> None:
        with open(opj(self.processed_dir, "config_info.json"), "r") as f:
            self.config = json.load(f)
        
        self.split_len=self.config[self.split]
        self.distribution = self.config["distribution"]

    def process(self):
        global TOTAL_FRAME_COUNT


        if self.fix_null:
            self._fix_null()

        train_dir, test_dir, val_dir = opj(self.processed_dir, "train"), opj(self.processed_dir, "test"), opj(self.processed_dir, "val")
        
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(val_dir)

        derived_dir = opj(self.raw_dir, "bimacs_derived_data")
        gt_dir = opj(self.raw_dir, "bimacs_rgbd_data_ground_truth")

        with open(opj(self.raw_dir, "BimanualActions", "bimacs_video_id_to_video_fps.json"), "r") as f:
            self.fps_info = json.load(f)

        train_sample_idx=0
        val_sample_idx=0
        test_sample_idx=0

        imputation_count=0
        prepended_things_count=0

        progressbar_manager=enlighten.get_manager()
        ticks=progressbar_manager.counter(total=540, desc="Progress:", unit="folder", color="blue")
        
        test_idx_sample_mapping = {} # aslında list de olur ama ne olur ne olmaz

        # To double check
        train_sample_count = 0
        val_sample_count = 0
        test_sample_count = 0

        for subject, task, take in crawl_dataset():
            
            # Sometimes initial graphs are invalid, so imputation is not working. This is to fix that
            prepend_count=0

            # make sure there is no cross-folder graph sequence
            data_list = [] 
            data_list_mirrored = [] 

            object_dir = opj(derived_dir, subject, task, take, "3d_objects")
            object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

            relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
            relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))

            gt_action_json = opj(gt_dir, subject, task, f"{take}.json")
            act_list=_get_actions_list(gt_action_json, len(relation_json_list))

            assert len(relation_json_list) == len(object_json_list)

            # To double check
            if int(subject[-1])==self.test_subject:
                test_sample_count = test_sample_count + len(relation_json_list) - self.temporal_length + 1
            elif int(take[-1])==self.validation_take:
                val_sample_count = val_sample_count + len(relation_json_list) - self.temporal_length + 1
            else:
                train_sample_count = train_sample_count + len(relation_json_list) - self.temporal_length + 1


            for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):
                
                TOTAL_FRAME_COUNT+=1
                graph, graph_mirrored = _get_graph(relation_json, object_json, act_list[frame], frame, subject, task, take)

                # If the graph is not valid, try to fix it (impute maybe). 
                # If it is not possible resets the sequence and continues
                validity = _check_validity(graph)
                
                if validity == "valid":
                    pass
                elif validity in ["null"]:
                    raise Exception(f"Unhandled graph error: {validity}")
                elif len(data_list) > 0:
                    # use previous graph to impute
                    graph = _impute_with_previous(graph, data_list[-1], validity)
                    graph_mirrored = _impute_with_previous(graph_mirrored, data_list_mirrored[-1], validity)
                    imputation_count+=1
                else:
                    # print("Skipping: invalid graph and no previous graph to impute: ", subject, task, take, "f", frame)
                    prepend_count+=1
                    prepended_things_count+=1
                    # data_list=[]
                    # data_list_mirrored=[]
                    continue
                
                data_list.append(graph)    
                data_list_mirrored.append(graph_mirrored)
                
                # TODO: Fix_fps and prepend_count are not compatible. Fix it
                if prepend_count != 0 and len(data_list) == 1:
                    for _ in range(prepend_count):
                        data_list.append(graph.clone())
                        data_list_mirrored.append(graph_mirrored.clone())
                    assert len(data_list) == prepend_count+1
                    prepend_count=0

                # TODO: we pop only one graph, but when fix_fps is true, we need to pop two of them
                # Check if fps need to be fixed. If so, add same graph again  
                if self.fix_fps and self.fps_info[subject+"-"+task+"-"+take] == 15:
                    data_list.append(graph)
                    data_list_mirrored.append(graph_mirrored)
                    
                # if the sequence is complete, saves it
                if len(data_list) == self.temporal_length:
                    
                    if int(subject[-1])==self.test_subject:
                        torch.save(data_list, opj(test_dir, f'data_{test_sample_idx}.pt'))
                        test_idx_sample_mapping[test_sample_idx] = (subject, task, take, frame-self.temporal_length, frame, "not_mirrored")
                        test_sample_idx+=1

                        torch.save(data_list_mirrored, opj(test_dir, f'data_{test_sample_idx}.pt'))
                        test_idx_sample_mapping[test_sample_idx] = (subject, task, take, frame-self.temporal_length, frame, "mirrored")
                        test_sample_idx+=1
                    
                    elif int(take[-1])==self.validation_take:
                        torch.save(data_list, opj(val_dir, f'data_{val_sample_idx}.pt'))
                        val_sample_idx+=1
                        
                        torch.save(data_list_mirrored, opj(val_dir, f'data_{val_sample_idx}.pt'))
                        val_sample_idx+=1

                    else:
                        torch.save(data_list, opj(train_dir, f'data_{train_sample_idx}.pt'))
                        train_sample_idx+=1

                        torch.save(data_list_mirrored, opj(train_dir, f'data_{train_sample_idx}.pt'))
                        train_sample_idx+=1

                    self.distribution = self.distribution + data_list[0].y[0] + data_list[0].y[1]
                    data_list[0].validate(raise_on_error=True)
                    data_list_mirrored[0].validate(raise_on_error=True)
                    data_list.pop(0)
                    data_list_mirrored.pop(0)

            ticks.update()


        progressbar_manager.stop()
        
        assert train_sample_idx == train_sample_count*2, f"{train_sample_idx=} ,{train_sample_count*2=}"
        assert val_sample_idx == val_sample_count*2, f"{val_sample_idx=} ,{val_sample_count*2=}"
        assert test_sample_idx == test_sample_count*2, f"{test_sample_idx=} ,{test_sample_count*2=}"

        print(f"\n\n#######\n{TOTAL_FRAME_COUNT=}, {EMPTY_COUNT=}, {NO_EDGE_COUNT=}, {MISSING_HAND_COUNT=}, {NULL_COUNT=}")
        
        self.config = {"test": test_sample_idx, 
                       "train": train_sample_idx, 
                       "val": val_sample_idx, 
                       "test_subject": self.test_subject, 
                       "validation_take": self.validation_take,
                       "temporal_length": self.temporal_length,
                       "distribution": self.distribution.tolist()}
        
        print(self.config)
        print(f"{imputation_count=}, {prepended_things_count=}")
        
        self._save_config(self.config)
        
        with open(opj(self.processed_dir, "test_idx_sample_mapping.json"), "w") as f:
            json.dump(test_idx_sample_mapping, f)
        
        self.is_processed=True

    def len(self):
        # return len(self.processed_file_names)
        return self.split_len

    def get(self, idx):
        # data = torch.load(opj(self.processed_dir, f'data_{idx}.pt'))
        # f=glob.glob(opj(self.processed_dir, f'data*_{idx}.pt'))
        # data = torch.load(f[0])

        data = torch.load(opj(self.processed_dir, self.split, f'data_{idx}.pt'))
        return idx, data


# Simple test
def print_data_point(data: Data):

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
    print("Left:", ACTIONS[torch.argmax(data.y[0, :-1], dim=0)], ", Right: ", ACTIONS[torch.argmax(data.y[1, :-1], dim=0)])

    print("Graph id: ", data.y[0,-1].item())


def get_bimanual_dataset_splits(*, root: str, is_processed: bool, test_subject: int, validation_take: int , temporal_length: int, contact_only: bool, fix_fps: bool, fix_null: bool) -> Tuple[Dataset, Dataset, Dataset]:
    # Wrapper function to prepare dataset splits (train, test, validation)
    
    train_set=BimanualDataset(root, is_processed, "train", test_subject, validation_take, temporal_length, contact_only, fix_fps, fix_null)
    test_set=BimanualDataset(root, True, "test", test_subject, validation_take, temporal_length, contact_only, fix_fps)
    val_set=BimanualDataset(root, True, "val", test_subject, validation_take, temporal_length, contact_only, fix_fps)
    
    print("Datasets are ready")
    return train_set, test_set, val_set

# bm = BimanualDataset(root=".", split="train",  test_subject=1, validation_take=7, is_processed=True, temporal_length=10)

# # train_pseudo=bm[500:]
# # test_pseudo=bm[:500]

# loader = DataLoader(bm, batch_size=4, shuffle=True)
# batch=next(iter(loader))

# import numpy as np
# all=np.zeros((10,4))

# for i, b in enumerate(batch):
#     # print("\n\n#### batch ####")
#     for j, g in enumerate(b.to_data_list()):
#         # print("#### gs ####")
#         # print(g.y[0,-1].item())
#         all[i, j]=g.y[0,-1].item()

# print(all)