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

        # self.pred_left=Sequential(Linear(hidden_size, num_action_classes), Softmax(dim=1))
        # self.pred_right=Sequential(Linear(hidden_size, num_action_classes), Softmax(dim=1))
        self.mlp=Sequential(Linear(hidden_size, num_action_classes), Softmax(dim=1))

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
            
            H1 = self.act1(self.conv1(x, edge_index, edge_attr)) # x dim -> [n, d]    
            H2 = self.act2(self.conv2(H1, edge_index, edge_attr)) # x dim -> [n, d]    

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
