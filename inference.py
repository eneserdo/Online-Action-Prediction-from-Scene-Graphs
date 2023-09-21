# To be runned on xg environment

import numpy as np
from utils import arg_parser_inference
import models
from dataset import OBJECTS, RELATIONS, ACTIONS
import torch
from os.path import join as opj
from torch_geometric.data import Data

import rospy
from rospy.numpy_msg import numpy_msg

class RosRun:
    def __init__(self):
        """
        Example Usage:
        
        """


        self.rgb_topic = "/zed/zed_node/rgb/image_rect_color"
        self.depth_topic = "/zed/zed_node/depth/depth_registered"
        

        self.temp_rgb = None
        self.temp_depth = None
        self.rgb_ready = False
        self.depth_ready = False
        # self.br=CvBridge()

    
    def imgCallback1(self, data):
        self.temp_rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        
        # self.temp_img1 = self.br.imgmsg_to_cv2(data, "passthrough")    
        self.rgb_ready = True


    def frameGenerator(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Img_Msg, self.imgCallback1)
        rospy.Subscriber(self.depth_topic, Img_Msg, self.imgCallback2)
        print("Waiting...")
        

        while not rospy.is_shutdown():       
            if self.rgb_ready and self.depth_ready:
                self.rgb_ready=False
                self.depth_ready=False
                yield self.temp_rgb, self.temp_depth
            
        
        yield None, None



def get_graph(obj_list, relation_tensor: np.ndarray):
    
    x = torch.zeros((len(obj_list), len(OBJECTS)))

    for i, obj in enumerate(obj_list):
        x[i, OBJECTS.index(obj["class_name"])] = 1
    
    edge_index
    edge_attr
    
    graph=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph


def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args=arg_parser_inference()
    
    model = models.GNN(num_edge_feat=len(RELATIONS),
                   conv_in=len(OBJECTS),
                   conv_hidden=args.conv_hidden,
                   conv_out=args.conv_out,
                   batch_size=args.batch_size,
                   num_action_classes=len(ACTIONS),
                   hidden_size=args.hidden_size,
                   num_rnn_layers=1).to(device=device)
    
    # restore the model
    checkpoint = torch.load(opj("saved", f"checkpoint_epoch{args.restore:0>4}.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("model and optimizer parameters are restored successfully!")
    
    print("# of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # load jsons from folder
    
    files=glob.glob()

    buffer=[]

    for f in files:
        g=get_graph()

        # returns none if not a valid graph (e.g. both hand must exist)
        if g is None:
            print("skipped graph")
            continue
        
        buffer.append(g)

        if len(buffer)==args.temporal_length:
            left_right_pred=model(buffer)
            buffer.pop(0)
            

    

if __name__ == '__main__':
    main()