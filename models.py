from typing import List
import torch
from torch.nn import Sequential, Linear, ReLU, SELU, LSTM, Softmax
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from dataset import OBJECTS

# TODO: graph-level -> node-level (hand-pooling, how to deal with missing-hand-graphs)
# TODO: Single MLP to learn both action simultaneousluy

class GNN(torch.nn.Module):
    def __init__(self,*, num_edge_feat, conv_in, conv_hidden, conv_out,  batch_size, num_action_classes,  hidden_size, num_rnn_layers):
        super().__init__()

        self.conv_out=conv_out
        self.batch_size=batch_size
        
        nn1 = Sequential(Linear(num_edge_feat, conv_in*conv_hidden), ReLU())
        self.conv1 = NNConv(conv_in, conv_hidden, nn1, aggr='mean')
        
        nn2 = Sequential(Linear(num_edge_feat, conv_hidden*conv_out), ReLU())
        self.conv2 = NNConv(conv_hidden, conv_out, nn2, aggr='mean')

        self.act1 = SELU()
        self.act2 = SELU()

        self.rnn = LSTM(conv_out, hidden_size, num_rnn_layers)

        self.pred_left=Sequential(Linear(hidden_size, num_action_classes), Softmax())
        self.pred_right=Sequential(Linear(hidden_size, num_action_classes), Softmax())

    def _hands_pooling(self, x, embedding, batch):
        left_hand=torch.zeros((1, len(OBJECTS)))
        left_hand[0, OBJECTS.index("LeftHand")]=1
  
        right_hand=torch.zeros((1, len(OBJECTS)))
        right_hand[0, OBJECTS.index("RightHand")]=1

        graph_hand_indicies=torch.zeros_like(batch)

        for i, obj in enumerate(x):    
            if obj==left_hand:
                graph_hand_indicies[i]=1

            if obj==right_hand:
                graph_hand_indicies[i]=2
        
        
        # left, right: [batch_size, len(embedding)]


    def forward(self, dataBatchList: List[Batch]):

        embedding_list=[]

        for databatch in dataBatchList:
            x, edge_attr, edge_index, batch  = databatch.x, databatch.edge_attr, databatch.edge_index, databatch.batch
            
            H1 = self.act1(self.conv1(x, edge_index, edge_attr)) # x dim -> [n, d]    
            H2 = self.act2(self.conv2(H1, edge_index, edge_attr)) # x dim -> [n, d]    

            graph_embedding = global_mean_pool(H2, batch) 
            # graph_embedding = self._hands_pooling(x=x, embedding=H2, batch=batch) 
            
            # assert graph_embedding.shape[0]==self.batch_size and graph_embedding.shape[1]==self.conv_out

            embedding_list.append(graph_embedding)

        sequencial_embeddings=torch.stack(embedding_list)
        # n_samples = y.size(0)
        # h_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)
        # c_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)

        y, (hn, cn) = self.rnn(sequencial_embeddings)
        # y[-1] is equal to hn (checked)

        return self.pred_left(hn), self.pred_right(hn)
