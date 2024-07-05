
from utils import arg_parser_inference
import models
from dataset import BimanualDataset, get_bimanual_dataset_splits, OBJECTS, RELATIONS, ACTIONS
import torch
from os.path import join as opj

def get_graph():
    # oradaki kodu kopyalayabilir miyiz
    return None

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