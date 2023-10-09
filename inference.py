# Bu kod inference_ros kodu ile birleştirilip tek bir kod haline getirilecek.

# Use get graph function in dataset
# Use buffer
# Use is_valid function
# Load model
# visualize and save result for whole video

import glob
import os
import numpy as np
import torch
import torch.nn as nn
from os.path import join as opj
from dataset import _get_graph, _check_validity, crawl_dataset_FULL, _get_actions

from utils import arg_parser_inference
from metrics import f1_at_k_single_example
import natsort
from visualization_utils import plot_segmentation
import models
from dataset import ACTIONS, OBJECTS, RELATIONS

# def majority_voting(y_pred_segmentation):
#     left_segment=y_pred_segmentation[:, 0, :]
#     right_segment=y_pred_segmentation[:, 1, :]
#     pass


def main(args):
    print(args)

    derived_dir = opj(args.raw_dir, "bimacs_derived_data")
    gt_dir = opj(args.raw_dir, "bimacs_rgbd_data_ground_truth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_subject = args.test_subject
    

    saved_dir = opj(args.saved_dir, args.restored_name)
    os.mkdir(saved_dir)
    
    # Load the model and restore the parameters
    model = models.ActionRecognition(gnn_type=args.gnn_type,
                                     temporal_type=args.temporal_type,
                                     num_edge_feat=len(RELATIONS),
                                     in_channels=len(OBJECTS),
                                     hidden_channels=args.hidden_channels,
                                     out_channels=args.out_channels,
                                     batch_size=args.batch_size,
                                     num_action_classes=len(ACTIONS),
                                     lstm_hidden_size=args.lstm_hidden_size,
                                     heads=args.num_heads,
                                     use_bn=not args.no_bn,
                                     filter_it=args.filter_idle_hold,
                                     use_hand_pooling=not args.use_global_pooling,
                                     edge_dropout=not args.edge_dropout,
                                     dropout=args.dropout,
                                     use_cls_token=args.use_cls_token).to(device=device)

    print(model)
    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    NULL_ACTION = len(ACTIONS)
    class_id_to_label = {i: label for i, label in enumerate(ACTIONS)}
    class_id_to_label[NULL_ACTION] = "NULL"

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
        
        f1_at_10_list = []
        f1_at_25_list = []
        f1_at_50_list = []

        y_pred = np.ones((len(relation_json_list), 2))*NULL_ACTION
        y_true = np.ones((len(relation_json_list), 2))*NULL_ACTION

        y_pred_segmentation = np.ones((len(relation_json_list), 2, len(relation_json_list)-args.temporal_lenght+1))*NULL_ACTION
        
        
        assert len(relation_json_list) == len(object_json_list)

        for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):
            
            TOTAL_FRAME_COUNT+=1
            graph, _ = _get_graph(relation_json, object_json, gt_action_json, frame, subject, task, take)
            
            left_gt, right_gt = _get_actions(gt_action_json, frame)
            y_true[frame, 0] = left_gt
            y_true[frame, 1] = right_gt

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
                
                # For frame-based evaluation
                y_pred[frame-args.temporal_length:frame, 0] = pred[0]
                y_pred[frame-args.temporal_length:frame, 1] = pred[1]
                
                # For segmentation evaluation
                # y_pred_segmentation[frame-args.temporal_length:frame, 0, frame+args.temporal_length] = pred[:, 0]

        
        y_pred = majority_voting(y_pred_segmentation)
        # TODO: visualize and save result for whole video
        
        segment_name=f"{subject}_{task}_{take}_"
        plot_save_dir = opj(saved_dir, segment_name)
        plot_segmentation(y_true[:, 0], y_pred[:, 0], class_id_to_label, plot_save_dir+"L.png", bar_height=50)
        plot_segmentation(y_true[:, 1], y_pred[:, 1], class_id_to_label, plot_save_dir+"R.png", bar_height=50)
        
        # calculate f1@k score
        # left
        f1_at_10_list.append(f1_at_k_single_example(y_true[:, 0].tolist(), y_pred[:, 0].tolist(), num_classes=len(ACTIONS)+1, overlap=0.1))
        f1_at_25_list.append(f1_at_k_single_example(y_true[:, 0].tolist(), y_pred[:, 0].tolist(), num_classes=len(ACTIONS)+1, overlap=0.25))
        f1_at_50_list.append(f1_at_k_single_example(y_true[:, 0].tolist(), y_pred[:, 0].tolist(), num_classes=len(ACTIONS)+1, overlap=0.5))
    
        # right
        f1_at_10_list.append(f1_at_k_single_example(y_true[:, 1].tolist(), y_pred[:, 1].tolist(), num_classes=len(ACTIONS)+1, overlap=0.1))
        f1_at_25_list.append(f1_at_k_single_example(y_true[:, 1].tolist(), y_pred[:, 1].tolist(), num_classes=len(ACTIONS)+1, overlap=0.25))
        f1_at_50_list.append(f1_at_k_single_example(y_true[:, 1].tolist(), y_pred[:, 1].tolist(), num_classes=len(ACTIONS)+1, overlap=0.5))
    
    # average the f1@k score over all videos
    f1_at_10 = np.mean(f1_at_10_list)
    f1_at_25 = np.mean(f1_at_25_list)
    f1_at_50 = np.mean(f1_at_50_list)


if __name__ == '__main__':
    main(arg_parser_inference.parse_args())