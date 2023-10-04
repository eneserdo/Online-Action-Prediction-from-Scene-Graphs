# To be runned on xg environment

from typing import List, Union
import numpy as np
# from utils import arg_parser_inference
# import models
# from dataset import OBJECTS, RELATIONS, ACTIONS
import torch
from os.path import join as opj
# from torch_geometric.data import Data

import rospy
from std_msgs.msg import Float64MultiArray

# from _Float64 import Float64
# from rospy.numpy_msg import numpy_msg


def deserialize_graph_raw(graph_serialized: List) -> Union[torch.Tensor, torch.Tensor]:
    """ num_obj, [objects], [flattened relation matrix] """
    num_obj  = int(graph_serialized[0])
    
    object_list = torch.tensor(graph_serialized[1:1+num_obj])
    # graph_tensor = torch.tensor(graph_serialized[1+num_obj:]).reshape(num_obj, num_obj, len(RELATIONS))
    graph_tensor = torch.tensor(graph_serialized[1+num_obj:]).reshape(num_obj, num_obj, 3) #TODO

    return object_list, graph_tensor


class RosRun:
    def __init__(self):
        self.data_topic = "/graph_raw"        
        self.data_ready = False

    def imgCallback(self, data):
        # self.graph_data = np.frombuffer(data.data, dtype=np.uint8)
        self.graph_data=data.data
        self.data_ready = True

    def frameGenerator(self):
        rospy.init_node('inference', anonymous=True)
        rospy.Subscriber(self.data_topic, Float64MultiArray, self.imgCallback)
        print("[inference] Waiting...")
        
        while not rospy.is_shutdown():       
            if self.data_ready:
                self.data_ready=False
                yield self.graph_data
            

# def get_graph(obj_list: torch.Tensor, relation_tensor: torch.Tensor) -> Data:
    
#     num_obj=obj_list.shape[0]
#     x = torch.zeros((num_obj, len(OBJECTS)))

#     for i, obj in enumerate(obj_list):
#         x[i, obj] = 1
    

#     edge_index = torch.zeros(num_obj*num_obj, 2) 
#     edge_attr = torch.zeros(num_obj*num_obj, len(RELATIONS)) 
#     c=0
#     for i in range(num_obj):
#         for j in range(num_obj):
#             if abs(relation_tensor[i, j].sum()-1) < 1e-5:
#                 edge_index[c,0]=obj_list[i]
#                 edge_index[c,1]=obj_list[j]
#                 edge_attr[c] = relation_tensor[i, j]
#                 c+=1

#     edge_index=edge_index[:c]
#     edge_attr=edge_attr[:c]

#     graph=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#     return graph

# def is_valid(data: Data) -> Union[Data | None, str]:
    
#     # null
#     if torch.sum(data.y[0])==0 or torch.sum(data.y[1])==0:
#         return None, "Null"

#     # not empty
#     if len(data.x)==0:
#         return None, "Empty"
    
#     # both hands
#     for row in data.x:
#         if row[OBJECTS.index("LeftHand")]==1:
#             Lhand=True
        
#         if row[OBJECTS.index("RightHand")]==1:
#             Rhand=True

#     if not (Rhand and Lhand):
#         return None, "Missing hand"

#     # has edge
#     if len(data.edge_index)==0:
#         return None, "No edge"
    
#     return data, "Valid"

def main():
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # args=arg_parser_inference()
    
    # model = models.ActionRecognition(gnn_type=args.gnn_type,
    #                                  temporal_type=args.temporal_type,
    #                                  num_edge_feat=len(RELATIONS),
    #                                  in_channels=len(OBJECTS),
    #                                  hidden_channels=args.hidden_channels,
    #                                  out_channels=args.out_channels,
    #                                  batch_size=args.batch_size,
    #                                  num_action_classes=len(ACTIONS),
    #                                  lstm_hidden_size=args.lstm_hidden_size,
    #                                  heads=args.num_heads,
    #                                  use_bn=not args.no_bn,
    #                                  filter_it=args.filter_idle_hold,
    #                                  use_hand_pooling=not args.use_global_pooling,
    #                                  edge_dropout=not args.edge_dropout,
    #                                  dropout=args.dropout,
    #                                  use_cls_token=args.use_cls_token).to(device=device)
    
    # # restore the model    
    # checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
    # model.load_state_dict(checkpoint["model_state_dict"])
    # print("model and optimizer parameters are restored successfully!")

    # print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # load jsons from folder
    # files=glob.glob()

    buffer=[]
    ros_runner=RosRun()
    i=0
    # for f in files:
    for graph_raw_serialized in ros_runner.frameGenerator():

        graph_raw = deserialize_graph_raw(graph_raw_serialized)

        # graph=get_graph(graph_raw)

        # # TODO: validity check as in dataset creation
        # graph, comment = is_valid(graph)
        # if graph == None:
        #     buffer=[]
        #     print("Skipped graph: ", comment)
        #     continue

        # graph = graph.to(device=device)
        
        # buffer.append(graph)

        # if len(buffer) == args.temporal_length:
        #     left_right_pred=model(buffer)
            
        #     left_right_pred = torch.argmax(left_right_pred, dim=1)
        #     print("L: ", ACTIONS[left_right_pred[0].item()], "R: ", ACTIONS[left_right_pred[1].item()])
        #     buffer.pop(0)
        # else:
        #     print("Buffer not full yet")
            
        i+=1
        print(i, graph_raw)

    

if __name__ == '__main__':
    main()