# Use get graph function in dataset
# Use buffer
# Use is_valid function
# Load model
# visualize and save result for whole video

# Color code:
# map 16 actions to distinct colors
action_colors = {   0: (0, 0, 255), 
                    1: (0, 255, 0), 
                    2: (255, 0, 0), 
                    3: (255, 255, 0), 
                    4: (255, 0, 255), 
                    5: (0, 255, 255), 
                    6: (255, 128, 0), 
                    7: (128, 255, 0), 
                    8: (0, 255, 128), 
                    9: (0, 128, 255), 
                    10: (128, 0, 255), 
                    11: (255, 0, 128), 
                    12: (128, 128, 0), 
                    13: (128, 0, 128), 
                    14: (0, 0, 0), # None  
                }

import glob
import os
import torch
import torch.nn as nn
from os.path import join as opj
from dataset import _get_graph, _check_validity, crawl_dataset_FULL

from utils import arg_parser_inference
from metrics import f1_at_k_single_example
import natsort
from visualization_utils import plot_segmentation


def main(args):
    print(args)

    derived_dir = opj(args.raw_dir, "bimacs_derived_data")
    gt_dir = opj(args.raw_dir, "bimacs_rgbd_data_ground_truth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_subject = args.test_subject
    

    saved_dir = opj(args.saved_dir, args.restored_name)
    os.mkdir(saved_dir)
    

    # Load the model and restore the parameters
    ...

    # Load data
    for subject, task, take in crawl_dataset_FULL():
        if subject != int(test_subject[-1]):
            continue

        # make sure there is no cross-folder graph sequence
        data_list = []  

        object_dir = opj(derived_dir, subject, task, take, "3d_objects")
        object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

        relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
        relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))

        gt_action_json = opj(gt_dir, subject, task, f"{take}.json")

        assert len(relation_json_list) == len(object_json_list)

        for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):
            
            TOTAL_FRAME_COUNT+=1
            graph, graph_mirrored = _get_graph(relation_json, object_json, gt_action_json, frame, subject, task, take)
            
            # if the graph is not appropriate, resets the sequence and continues
            if not _check_validity(graph):
                data_list=[]
                # print("Skipped: ", subject, task, take, "f", frame)
                continue

            data_list.append(graph)
            
            # if self.fix_fps and self.fps_info[subject+"-"+task+"-"+take] == 15:
                # data_list.append(graph)

            if len(data_list) == args.temporal_length:

                pred = model(data_list)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()

        # TODO: visualize and save result for whole video
        
        segment_name=f"{subject}_{task}_{take}.png"
        plot_save_dir = opj(saved_dir, segment_name)
        plot_segmentation()
        # calculate f1@k score

    # average the f1@k score over all videos



if __name__ == '__main__':
    main(arg_parser_inference.parse_args())