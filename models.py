from typing import List
import torch
from torch.nn import Sequential, Linear, ReLU, SELU, LSTM, Softmax
import torch_geometric.nn as geometric_nn
from torch_geometric.data import Data, Batch

from dataset import OBJECTS

# TODO: graph-level -> node-level (hand-pooling, how to deal with missing-hand-graphs)
# TODO: Single MLP to learn both action simultaneously

class ActionRecognition(torch.nn.Module):
    def __init__(self, gnn_type, in_channels, hidden_channels, out_channels, num_edge_feat, batch_size, num_action_classes, lstm_hidden_size, use_bn, **kwargs):
        """ Accepts kwargs for non-general params of gnns """
        super().__init__()

        # self.conv_out=out_channels
        self.batch_size=batch_size
        
        # Sets Graph Network
        if gnn_type == "NNConv":
            nn1 = Sequential(Linear(num_edge_feat, in_channels*hidden_channels), ReLU())
            self.conv1 = geometric_nn.NNConv(in_channels, hidden_channels, nn1, aggr='mean')
            nn2 = Sequential(Linear(num_edge_feat, hidden_channels*out_channels), ReLU())
            self.conv2 = geometric_nn.NNConv(hidden_channels, out_channels, nn2, aggr='mean')
            effective_out_ch1=hidden_channels
            effective_out_ch2=out_channels

        elif gnn_type == "GATConv":
            self.conv1 = geometric_nn.GATConv(in_channels, hidden_channels, kwargs["heads"], edge_dim=num_edge_feat )
            self.conv2 = geometric_nn.GATConv(hidden_channels*kwargs["heads"], out_channels, kwargs["heads"], edge_dim= num_edge_feat)
            effective_out_ch1=hidden_channels*kwargs["heads"]
            effective_out_ch2=out_channels*kwargs["heads"]

        elif gnn_type == "TransformerConv":
            self.conv1 = geometric_nn.TransformerConv(in_channels, hidden_channels, kwargs["heads"], edge_dim=num_edge_feat )
            self.conv2 = geometric_nn.TransformerConv(hidden_channels*kwargs["heads"], out_channels, kwargs["heads"], edge_dim= num_edge_feat)
            effective_out_ch1=hidden_channels*kwargs["heads"]
            effective_out_ch2=out_channels*kwargs["heads"]

        else:
            raise NotImplementedError(f"gnn_type: {gnn_type} is not implemented")

        self.act1 = SELU()
        self.act2 = SELU()


        if use_bn:
            self.bn1=geometric_nn.BatchNorm(effective_out_ch1)
            self.bn2=geometric_nn.BatchNorm(effective_out_ch2)
            self.gnn=geometric_nn.Sequential("x, edge_index, edge_attr", 
                                             [(self.conv1, "x, edge_index, edge_attr -> x"), self.act1, self.bn1, 
                                              (self.conv2, "x, edge_index, edge_attr -> x"), self.act2, self.bn2])
        else:
            self.gnn=geometric_nn.Sequential("x, edge_index, edge_attr", 
                                             [(self.conv1, "x, edge_index, edge_attr -> x"), self.act1, 
                                              (self.conv2, "x, edge_index, edge_attr -> x"), self.act2])


        num_rnn_layers = 1 # Lets make it constant
        self.rnn = LSTM(effective_out_ch2, lstm_hidden_size, num_rnn_layers)

        # self.pred_left=Sequential(Linear(hidden_size, num_action_classes), Softmax(dim=1))
        # self.pred_right=Sequential(Linear(hidden_size, num_action_classes), Softmax(dim=1))
        self.mlp=Sequential(Linear(lstm_hidden_size, num_action_classes), Softmax(dim=1))

    def _hands_pooling(self, x, embeddings, batch):
        
        labels=torch.argmax(x, dim=1)

        # left_hand=torch.zeros((1, len(OBJECTS)))
        # left_hand[0, OBJECTS.index("LeftHand")]=1
  
        # right_hand=torch.zeros((1, len(OBJECTS)))
        # right_hand[0, OBJECTS.index("RightHand")]=1

        graph_hand_indicies=torch.zeros((self.batch_size*2), dtype=torch.long)
        left_hand_indicies=torch.zeros((self.batch_size), dtype=torch.long)
        right_hand_indicies=torch.zeros((self.batch_size), dtype=torch.long)

        # for i, obj in enumerate(x):    
        #     if obj==left_hand:
        #         graph_hand_indicies[i]=1

        #     if obj==right_hand:
        #         graph_hand_indicies[i]=2
        

        for i, l in enumerate(labels):
            if l == OBJECTS.index("LeftHand"):
                # graph_hand_indicies[batch[i]*2+0]=i
                left_hand_indicies[batch[i]]=i
            elif l == OBJECTS.index("RightHand"):
                # graph_hand_indicies[batch[i]*2+1]=i
                right_hand_indicies[batch[i]]=i

        # left, right: [batch_size, len(embedding)]
        # return embeddings[graph_hand_indicies]
        return embeddings[left_hand_indicies], embeddings[right_hand_indicies]

    def forward(self, dataBatchList: List[Batch]):

        # embedding_list=[]
        left_embedding_list=[]
        right_embedding_list=[]

        for databatch in dataBatchList: # processes each frame time-independently
            x, edge_attr, edge_index, batch  = databatch.x, databatch.edge_attr, databatch.edge_index, databatch.batch
            
            # H1 = self.act1(self.conv1(x, edge_index, edge_attr)) # x dim -> [n, d]    
            # H2 = self.act2(self.conv2(H1, edge_index, edge_attr)) # x dim -> [n, d]    
            H2 = self.gnn(x, edge_index, edge_attr)

            left_embeddings, right_embeddings=self._hands_pooling(x,H2,batch) 
            # graph_embedding = global_mean_pool(H2, batch)
            
            # assert graph_embedding.shape[0]==self.batch_size and graph_embedding.shape[1]==self.conv_out

            # embedding_list.append(graph_embedding)
            left_embedding_list.append(left_embeddings)
            right_embedding_list.append(right_embeddings)

        # sequencial_embeddings=torch.stack(embedding_list)
        left_sequencial_embeddings=torch.stack(left_embedding_list)
        right_sequencial_embeddings=torch.stack(right_embedding_list)
        # n_samples = y.size(0)
        # h_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)
        # c_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)

        both_sequencial_embeddings=torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=1)

        # y, (hn, cn) = self.rnn(sequencial_embeddings)
        # y[-1] is equal to hn (checked)
        _, (both_hn, _) = self.rnn(both_sequencial_embeddings)
        assert both_hn.shape[0]==1
        both_hn=both_hn.squeeze()
        # _, (left_hn, _) = self.rnn(left_sequencial_embeddings)
        # _, (right_hn, _) = self.rnn(right_sequencial_embeddings)

        # return self.pred_left(hn), self.pred_right(hn)
        # return self.pred_left(left_hn), self.pred_right(right_hn)
        return self.mlp(both_hn)

    # def print_summary(self):
    # req pyg 2.3 
    #     x = torch.randn(100, 15) # num_node, inp_dim 
    #     edge_index = torch.randint(0, 20, size=(2, 20)) # 20 relation
    #     edge_attr = torch.randint(0, 2, size=(20,15)) # their attr
    #     batch = torch.randint(0, 5, size=(20,)) # their attr
    #     b=Batch(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)
    #     inp=[b, b, b, b]
    #     print(summary(self, inp))
