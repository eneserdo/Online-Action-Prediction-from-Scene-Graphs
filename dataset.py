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
import pickle


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


def _get_actions_list(gt_action_json: os.PathLike, act_len: int) -> np.ndarray:
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

    # graph_str=subject[-1]+task[5]+take[-1]+str(frame)

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
    

def download_bimanual_dataset(raw_dir: str, fix_null=True) -> None:
    # Downloads and unzip the Bimanual dataset to raw_dir.
    # Also fixes the null actions if fix_null is True (kind of a preprocessing)
    # TODO: move fix_fps to here

    urls = ["https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_3d_objects.zip",
            "https://bimanual-actions.humanoids.kit.edu/download/bimacs_derived_data_spatial_relations.zip",
            "https://bimanual-actions.humanoids.kit.edu/download/bimacs_rgbd_data_ground_truth.zip"]

    for url in urls:
        path = download_url(url, raw_dir)
        extract_zip(path, raw_dir)
    
    # Features https://drive.google.com/drive/folders/1yfwItIoQrAnbnk5GTjbbfN8Ls8Ybl_hr

    gdown.download(id="1FceipuZMUjp1ES2THqlAmQgPdE18-flk", output=opj(raw_dir, "BimanualActions.zip"))
    extract_zip(opj(raw_dir, "BimanualActions.zip"), raw_dir)


    def read_check_write(sub, task, tk_json, hand, new_vals):
        # An internal function to read, check and write the json file
        # Read
        the_json_path=opj(save_dir, "bimacs_rgbd_data_ground_truth", sub, task, tk_json)
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


    if fix_null:
        read_check_write("subject_6", "task_4_k_wiping", "take_1.json", "left_hand", 
                            [0, 0, 91, 1, 114, 3, 129, 5, 150, 4, 165, 2, 189, 1, 209, 5, 357])
        read_check_write("subject_1", "task_9_w_sawing", "take_2.json", "right_hand", 
                            [0, 0, 14, 1, 32, 3, 42, 9, 97, 9, 133])
        read_check_write("subject_1", "task_7_w_free_hard_drive", "take_7.json", "right_hand", 
                            [0, 0, 13, 1, 28, 3, 36, 2, 39, 1, 44, 3, 56, 11, 135, 2, 138, 1, 142, 3, 154, 0, 181, 1, 207, 11, 285, 2, 290, 1, 292, 3, 300, 4, 309, 4, 331, 2, 333, 1, 337, 4, 350, 2, 358, 0, 381])
        read_check_write("subject_3", "task_2_k_cooking_with_bowls", "take_1.json", "right_hand", 
                            [0, 0, 61, 1, 85, 5, 194, 1, 199, 1, 206, 3, 241, 6, 339, 4, 366, 2, 371, 1, 376, 5, 475, 2, 521, 0, 542])

        print("# Replaced null actions by manually detected labels (This is directly done on the raw data!)")

def _get_max_length(root: str, downsample) -> int:
    # Returns the maximum length of the videos (used for padding)
    max_length = 0
    for subject, task, take in crawl_dataset():
        relation_dir = opj(root, "raw", "bimacs_derived_data", subject, task, take, "spatial_relations")
        curr_list = glob.glob(opj(relation_dir, "*.json"))[::downsample]
        curr_length = len(curr_list)
        max_length = max(max_length, curr_length)
    return max_length

def __get_padding_graph() -> Data:
    # Returns a dummy graph for padding
    # To separate dummy from real check x.sum() == -1 or y.sum() == -1
    x = - torch.ones((1, 1), dtype=torch.float)
    edge_index = torch.tensor([[0, 0]], dtype=torch.long)
    edge_attr = torch.zeros((1, 1), dtype=torch.float)
    y = torch.zeros((2, len(ACTIONS)), dtype=torch.float)
    y[0, 0] = -1
    return Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)

def _pad_sequence(data_list: List[Data], target_length: int) -> List[Data]:
    # Pads the sequence to target_length
    # If the sequence is already longer than target_length, it is not padded
    # If the sequence is shorter than target_length, it is padded with a dummy graph
    if len(data_list) < target_length:
        for _ in range(target_length - len(data_list)):
            data_list.append(__get_padding_graph())

def process_bimanual_dataset(root: str, temporal_length: int, fix_fps: bool=False, downsample: int = 1) -> None:
    global TOTAL_FRAME_COUNT

    ds_append=f"_d{downsample}" if downsample > 1 else ""
    
    if temporal_length == -1:
        # whole video as one sample
        processed_dir = opj(root, "processed" + "Whole" + ds_append)
        process_whole_video = True
        MAX_VID_LENGTH = _get_max_length(root, downsample)
    else:
        processed_dir = opj(root, "processed"+str(temporal_length) + ds_append)
        process_whole_video = False

    
    os.mkdir(processed_dir)

    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        os.mkdir(opj(processed_dir, subject))


    raw_dir = opj(root, "raw")
    derived_dir = opj(raw_dir, "bimacs_derived_data")
    gt_dir = opj(raw_dir, "bimacs_rgbd_data_ground_truth")

    with open(opj(raw_dir, "BimanualActions", "bimacs_video_id_to_video_fps.json"), "r") as f:
        fps_info = json.load(f)

    sample_idx=0

    imputation_count=0
    prepended_things_count=0

    dataset_mapping_master = {"sample":{}, "vid":{}}

    # To double check
    sample_count = 0

    # Progress bar
    progressbar_manager=enlighten.get_manager()
    ticks=progressbar_manager.counter(total=540, desc="Progress:", unit="folder", color="blue")

    for subject, task, take in crawl_dataset():

        subject_int, task_int, take_int = int(subject[-1]), int(task[5]), int(take[-1])
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
        assert len(relation_json_list) == len(act_list)

        # Just to make sure at least one sample is saved per video. 
        # Mostly used for debugging whole video processing
        successful_save = False
        if process_whole_video:
            temporal_length = len(object_json_list[::downsample])
            is_padding_needed = False


        ds_act_list = act_list[::downsample]
        dataset_mapping_master["vid"][str(subject_int)+str(task_int)+str(take_int)] = ds_act_list
        sample_count = sample_count + len(ds_act_list) - temporal_length + 1 # This is redundant for whole video processing   

        for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):

            TOTAL_FRAME_COUNT += 1
            graph, graph_mirrored = _get_graph(relation_json, object_json, act_list[frame], frame, subject, task, take)

            ### Prepending and imputation ###
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
            ##################################


            # TODO: we pop only one graph, but when fix_fps is true, we need to pop two of them
            # Check if fps need to be fixed. If so, add same graph again  
            # if fix_fps and fps_info[subject+"-"+task+"-"+take] == 15:
            #     data_list.append(graph)
            #     data_list_mirrored.append(graph_mirrored)
                
            # if the sequence is complete, saves it
            if len(data_list) == temporal_length:
                
                # Am i genius or what
                data_list = data_list[::downsample]
                data_list_mirrored = data_list_mirrored[::downsample]

                # There is already a collate function for DataLoader in pyg, 
                # so it is not trivial to use DataLoader for this purpose. Instead, 
                # I padded manually.
                # Padding
                if process_whole_video:
                    # if MAX_VID_LENGTH - len(data_list) > 0:
                    #     is_padding_needed = True
                    #     for _ in range(MAX_VID_LENGTH - len(data_list)):
                    #         data_list.append(_get_padding_graph())
                    #         data_list_mirrored.append(_get_padding_graph())
                    _pad_sequence(data_list, MAX_VID_LENGTH)
                    _pad_sequence(data_list_mirrored, MAX_VID_LENGTH)


                # To make sure no padding goes into training 
                # TODO: will be removed
                if process_whole_video:
                    last_element_idx = len(dataset_mapping_master["vid"][str(subject_int)+str(task_int)+str(take_int)]) - 1
                    assert data_list[last_element_idx].y.sum() == 2, "WTF"
                    assert data_list_mirrored[last_element_idx].y.sum() == 2, "WTF"
                    
                    if is_padding_needed:
                        # make sure last element is not padding
                        assert data_list[last_element_idx + 1].y.sum() == -1, "WTF"
                        assert data_list_mirrored[last_element_idx + 1].y.sum() == -1, "WTF"

    
                torch.save(data_list, opj(processed_dir, subject, f'data_{sample_idx}.pt'))
                dataset_mapping_master["sample"][sample_idx] = (subject_int, task_int, take_int, frame-temporal_length + 1, frame + 1, "_")
                sample_idx+=1

                dataset_mapping_master["sample"][sample_idx] = (subject_int, task_int, take_int, frame-temporal_length + 1, frame + 1, "M")
                torch.save(data_list_mirrored, opj(processed_dir, subject, f'data_{sample_idx}.pt'))
                sample_idx+=1

                successful_save = True
                data_list[0].validate(raise_on_error=True)
                data_list_mirrored[0].validate(raise_on_error=True)
                data_list.pop(0)
                data_list_mirrored.pop(0)
        # End of for loop (frame)
                
        if not successful_save:
            raise ValueError("WTF")

        ticks.update()

    # End of for loop (video)


    progressbar_manager.stop()
    
    assert sample_idx == sample_count*2, f"{sample_idx=} ,{sample_count*2=}"

    print(f"\n\n#######\n{TOTAL_FRAME_COUNT=}, {EMPTY_COUNT=}, {NO_EDGE_COUNT=}, {MISSING_HAND_COUNT=}, {NULL_COUNT=}")
    
    
    print(f"{imputation_count=}, {prepended_things_count=}")

    config_info = {"sample_count": sample_idx, 
                   "temporal_length": temporal_length,
                   "fix_fps": fix_fps,
                   "downsample": downsample}

    with open(opj(processed_dir, "config_info_master.json"), "w") as f:
        json.dump(config_info, f)
        
    with open(opj(processed_dir, "dataset_mapping_master.pickle"), "wb") as f:
        pickle.dump(dataset_mapping_master, f)


class BimanualDataset(Dataset):
    def __init__(self, root, temporal_length, mapping):

        self.temporal_length = temporal_length

        self.pdir = opj(root, "processed"+str(temporal_length))
        
        self.mapping = mapping

        self.split_len = len(mapping)
        
        self.distribution = torch.zeros((len(ACTIONS)))

        transform, pre_transform, pre_filter = None, None, None
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_dir(self) -> str:
        return self.pdir

    @property
    def raw_file_names(self):
        return ["bimacs_rgbd_data_ground_truth/subject_1/task_1_k_cooking/take_0.json"]

    @property
    def processed_file_names(self):
        addr = [opj("subject_1", 'data_0.pt')] # TODO: maybe change this
        return addr


    def len(self) -> int:
        return self.split_len

    def get(self, idx: int) -> Data:

        path, global_idx = self.mapping[idx]
        data = torch.load(opj(self.processed_dir, path))

        # TODO maybe return both global_idx and local_idx
        return global_idx, data 



def create_mappings(processed_dir: str, temporal_length: int, test_subject: int, validation_take: int) -> [dict, dict]:
    
    path_mapping = {"train": {}, "test": {}, "val": {}} # [split]: local idx -> path, global idx 

    dataset_mapping = {"train":{"vid":{}, "sample":{}},
                       "test":{"vid":{}, "sample":{}},
                       "val":{"vid":{}, "sample":{}}}
    
    with open(opj(processed_dir, "config_info_master.json"), "r") as f:
        config_master = json.load(f)


    with open(opj(processed_dir, "dataset_mapping_master.pickle"), "rb") as f:
        dataset_mapping_master = pickle.load(f)


    # vids
    # dataset_mapping_master["vid"][str(subject_int)+str(task_int)+str(take_int)] = act_list
    for id, act_list in dataset_mapping_master["vid"].items():

        subject_int, task_int, take_int = int(id[0]), int(id[1]), int(id[2])

        if subject_int == test_subject:
            dataset_mapping["test"]["vid"][id] = act_list
        elif take_int == validation_take:
            dataset_mapping["val"]["vid"][id] = act_list
        else:
            dataset_mapping["train"]["vid"][id] = act_list


    # samples
    test_idx = 0
    val_idx = 0
    train_idx = 0

    # dataset_mapping_master["sample"][sample_idx] = (subject_int, task_int, take_int, frame-temporal_length + 1, frame + 1, "_" or "M") 
    for idx, (sub, task, take, start, end, mirrored) in dataset_mapping_master["sample"].items():
        
        if sub == test_subject:
            dataset_mapping["test"]["sample"][idx] = (sub, task, take, start, end, mirrored)
            path_mapping["test"][test_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            test_idx+=1
        
        elif take == validation_take:
            dataset_mapping["val"]["sample"][idx] = (sub, task, take, start, end, mirrored)
            path_mapping["val"][val_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            val_idx+=1

        else:
            dataset_mapping["train"]["sample"][idx] = (sub, task, take, start, end, mirrored)
            path_mapping["train"][train_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            train_idx+=1

    
    assert train_idx + test_idx + val_idx == config_master["sample_count"], f"{train_idx + test_idx + val_idx=}, {config_master['sample_count']=}" 


    return dataset_mapping, path_mapping


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


def get_bimanual_dataset_splits(*, root: str, process_it_anyway: bool, test_subject: int, validation_take: int , temporal_length: int, downsample: int) -> Tuple[Dataset, Dataset, Dataset]:
    # dataset_mapping = {"train":{"vid":{}, "sample":{}},
    #                    "test":{"vid":{}, "sample":{}},
    #                    "val":{"vid":{}, "sample":{}}}
    
    # path_mapping = {"train": {}, "test": {}, "val": {}} # [split]: idx -> path

    ds_append=f"_d{downsample}" if downsample > 1 else ""

    if temporal_length == -1:
        # whole video as one sample
        processed_dir = opj(root, "processed" + "Whole" + ds_append)
    else:
        processed_dir = opj(root, "processed"+str(temporal_length) + ds_append)

    
    assert os.path.isfile(opj(processed_dir, "dataset_mapping_master.pickle")), "Dataset is not preprocessed yet"
    assert os.path.isfile(opj(processed_dir, "config_info_master.json")), "Dataset is not preprocessed yet"

    if os.path.exists(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle")) and (not process_it_anyway):
        # read from folder
        print("Reading dataset mappings...")
        with open(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle"), "rb") as f:
            dataset_mapping = pickle.load(f)
        
        with open(opj(processed_dir, f"path_mapping_fold{test_subject}.pickle"), "rb") as f:
            path_mapping = pickle.load(f)

    else:
        print("Creating dataset mappings...")

        dataset_mapping, path_mapping = create_mappings(processed_dir, temporal_length, test_subject, validation_take)

        with open(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle"), "wb") as f:
            pickle.dump(dataset_mapping, f)

        with open(opj(processed_dir, f"path_mapping_fold{test_subject}.pickle"), "wb") as f:
            pickle.dump(path_mapping, f)


    train_set=BimanualDataset(root, temporal_length, path_mapping["train"])
    test_set=BimanualDataset(root, temporal_length, path_mapping["test"])
    val_set=BimanualDataset(root, temporal_length, path_mapping["val"])
    

    return train_set, test_set, val_set
