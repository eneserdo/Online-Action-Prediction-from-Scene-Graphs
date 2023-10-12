# Bu kod inference_ros kodu ile birleştirilip tek bir kod haline getirilecek.

# Use get graph function in dataset
# Use buffer
# Use is_valid function
# Load model
# visualize and save result for whole video

import glob
import os
import torch
import torch.nn as nn
from os.path import join as opj
from dataset import _get_graph, _check_validity, crawl_dataset_FULL, _get_actions_list

from torch_geometric.data import Data, Batch
from torchmetrics.classification import MulticlassF1Score

from utils import arg_parser_inference
from metrics import f1_at_k_single_example
import natsort
from visualization_utils import plot_segmentation
import models
from dataset import ACTIONS, OBJECTS, RELATIONS

def main(args):
    print(args)

    derived_dir = opj(args.raw_dir, "bimacs_derived_data")
    gt_dir = opj(args.raw_dir, "bimacs_rgbd_data_ground_truth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_subject = args.test_subject
    

    save_dir = opj(args.save_dir, args.restored_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model and restore the parameters
    model = models.ActionRecognition(gnn_type=args.gnn_type,
                                     temporal_type=args.temporal_type,
                                     num_edge_feat=len(RELATIONS),
                                     in_channels=len(OBJECTS),
                                     hidden_channels=args.hidden_channels,
                                     out_channels=args.out_channels,
                                     batch_size=1,
                                     num_action_classes=len(ACTIONS),
                                     lstm_hidden_size=args.lstm_hidden_size,
                                     heads=args.num_heads,
                                     use_bn=not args.no_bn,
                                     filter_it=args.filter_idle_hold,
                                     use_hand_pooling=not args.use_global_pooling,
                                     edge_dropout=not args.edge_dropout,
                                     dropout=args.dropout,
                                     use_cls_token=args.use_cls_token).to(device=device)

    # TODO: load model
    checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("model parameters are restored successfully!")
    
    # print(model)
    # print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.eval()

    NULL_ACTION = len(ACTIONS)
    class_id_to_label = {i: label for i, label in enumerate(ACTIONS)}
    class_id_to_label[NULL_ACTION] = "NULL"


    f1_at_10_list = []
    f1_at_25_list = []
    f1_at_50_list = []
    i=0
    # TODO: double check the metric
    f1_score=MulticlassF1Score(num_classes=len(ACTIONS)+1, average="macro")

    for subject, task, take in crawl_dataset_FULL():
        if test_subject != int(subject[-1]):
            continue
        # i+=1
        # if i==5:
        #     break

        # make sure there is no cross-folder graph sequence
        data_list = []

        object_dir = opj(derived_dir, subject, task, take, "3d_objects")
        object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

        relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
        relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))
        assert len(relation_json_list) == len(object_json_list)

        gt_action_json = opj(gt_dir, subject, task, f"{take}.json")

        y_pred = np.ones((len(relation_json_list), 2))*NULL_ACTION
        y_true = np.ones((len(relation_json_list), 2))*NULL_ACTION
        
        # y_pred_segmentation = np.ones((len(relation_json_list), 2, len(relation_json_list)-args.temporal_length+1))*NULL_ACTION
        act_list = _get_actions_list(gt_action_json, len(relation_json_list))

        for frame, (relation_json, object_json) in enumerate(zip(relation_json_list, object_json_list)):

            graph, _ = _get_graph(relation_json, object_json, gt_action_json, frame, subject, task, take)
            # TODO: mirrored graph

            graph = Batch.from_data_list([graph])

            graph=graph.to(device)

            left_gt, right_gt = act_list[frame]

            # if left_gt == None or right_gt == None:
            #     print("None action: ", subject, task, take, "f", frame, "  L:", left_gt, "R:", right_gt)

            y_true[frame, 0] = left_gt
            y_true[frame, 1] = right_gt
            # y_true_L.append(left_gt)
            # y_true_R.append(right_gt)


            # if the graph is not appropriate, resets the sequence and continues
            validity, explanation = _check_validity(graph)
            if not validity:
                data_list=[]
                # y_pred_L.append(NULL_ACTION)
                # y_pred_R.append(NULL_ACTION)

                print("Skipped: ", subject, task, take, "f", frame, "E:", explanation)
                continue

            data_list.append(graph)
            
            # if self.fix_fps and self.fps_info[subject+"-"+task+"-"+take] == 15:
                # data_list.append(graph)

            if len(data_list) == args.temporal_length:
                
                pred = model(data_list)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()
                
                # For frame-based evaluation
                # y_pred[frame-args.temporal_length:frame, 0] = pred[0]
                # y_pred[frame-args.temporal_length:frame, 1] = pred[1]
                y_pred[frame, 0] = pred[0]
                y_pred[frame, 1] = pred[1]

                # y_pred_L.append(pred[0])
                # y_pred_R.append(pred[1])
                
                # For segmentation evaluation
                # y_pred_segmentation[frame-args.temporal_length:frame, 0, frame+args.temporal_length] = pred[:, 0]
            
                data_list.pop(0)
                
        # y_pred = majority_voting(y_pred_segmentation)
        # TODO: visualize and save result for whole video
        
        segment_name=f"{subject}_{task}_{take}_"
        plot_save_dir = opj(save_dir, segment_name)
        
        y_true=y_true.astype(np.int32)
        y_pred=y_pred.astype(np.int32)

        assert (y_pred[:, 1]==NULL_ACTION).sum() == (y_pred[:, 0]==NULL_ACTION).sum(), "Left and right action should have same number of NULL actions"

        mask_to_keep = y_pred[:, 1]!=NULL_ACTION
        print("masked ratio: ", (1-mask_to_keep).sum()/len(mask_to_keep), flush=True)

        y_true=y_true[mask_to_keep]
        y_pred=y_pred[mask_to_keep]
        
        y_true_L, y_true_R =  y_true[:, 0], y_true[:, 1]
        y_pred_L, y_pred_R =  y_pred[:, 0], y_pred[:, 1]

        # plot_segmentation(y_true[:, 0], y_pred[:, 0], class_id_to_label=class_id_to_label, save_file=plot_save_dir+"L.png", bar_height=50)
        # plot_segmentation(y_true[:, 1], y_pred[:, 1], class_id_to_label=class_id_to_label, save_file=plot_save_dir+"R.png", bar_height=50)
        
        plot_segmentation(y_true_L, y_pred_L, class_id_to_label=class_id_to_label, save_file=plot_save_dir+"L.png", bar_height=50)
        plot_segmentation(y_true_R, y_pred_R, class_id_to_label=class_id_to_label, save_file=plot_save_dir+"R.png", bar_height=50)
        
        
        # calculate f1@k score
        # left
        f1_at_10_list.append(f1_at_k_single_example(y_true_L, y_pred_L, num_classes=len(ACTIONS)+1, overlap=0.1))
        f1_at_25_list.append(f1_at_k_single_example(y_true_L, y_pred_L, num_classes=len(ACTIONS)+1, overlap=0.25))
        f1_at_50_list.append(f1_at_k_single_example(y_true_L, y_pred_L, num_classes=len(ACTIONS)+1, overlap=0.5))
    
        # right
        f1_at_10_list.append(f1_at_k_single_example(y_true_R, y_pred_R, num_classes=len(ACTIONS)+1, overlap=0.1))
        f1_at_25_list.append(f1_at_k_single_example(y_true_R, y_pred_R, num_classes=len(ACTIONS)+1, overlap=0.25))
        f1_at_50_list.append(f1_at_k_single_example(y_true_R, y_pred_R, num_classes=len(ACTIONS)+1, overlap=0.5))
        
        # calculate regular f1 score
        f1_score(torch.tensor(y_pred_L), torch.tensor(y_true_L))
        f1_score(torch.tensor(y_pred_R), torch.tensor(y_true_R))

    # average the f1@k score over all videos
    f1_at_10 = sum(f1_at_10_list)/len(f1_at_10_list)
    f1_at_25 = sum(f1_at_25_list)/len(f1_at_25_list)
    f1_at_50 = sum(f1_at_50_list)/len(f1_at_50_list)

    f1=f1_score.compute()

    print(f"f1@10: {f1_at_10}, f1@25: {f1_at_25}, f1@50: {f1_at_50}")
    print(f"regular f1 score: {f1}")

    with open(opj(save_dir, "f1_at_k.txt"), "w") as f:
        f.write(f"f1@10: {f1_at_10}, f1@25: {f1_at_25}, f1@50: {f1_at_50}")
        f.write(f"regular f1 score: {f1}")

if __name__ == '__main__':
    main(arg_parser_inference())