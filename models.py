from typing import List
import torch
from torch.nn import Sequential, Linear, ReLU, SELU, LSTM, Softmax
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# TODO: graph-level -> node-level

class GNN(torch.nn.Module):
    def __init__(self, edge_linear_in, edge_linear_out, conv_in, conv_out,  batch_size, num_action_classes=15,  hidden_size=1, num_rnn_layers=1):
        super().__init__()

        self.conv_out=conv_out
        self.batch_size=batch_size
        nn = Sequential(Linear(edge_linear_in, edge_linear_out), ReLU())
        self.conv = NNConv(conv_in, conv_out, nn, aggr='mean')

        self.act = SELU()

        self.rnn = LSTM(conv_out, hidden_size, num_rnn_layers, batch_first=True)

        self.pred=Sequential(Linear(hidden_size, num_action_classes), Softmax())

    def forward(self, dataBatchList: List[Batch]):

        embedding_list=[]

        for databatch in dataBatchList:
            x, edge_attr, edge_index, batch  = databatch.x, databatch.edge_attr, databatch.edge_index, databatch.batch
            H = self.act(self.conv(x, edge_index, edge_attr)) # x dim -> [n, d]    
            # graph_embedding = torch.mean(H, dim=0, keepdim=True)
            graph_embedding = global_mean_pool(H, batch) 
            
            assert graph_embedding.shape[0]==self.batch_size and graph_embedding.shape[1]==self.conv_out

            embedding_list.append(graph_embedding)

        sequencial_embeddings=torch.stack(embedding_list)
        # n_samples = y.size(0)
        # h_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)
        # c_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)

        y, (hn, cn) = self.rnn(sequencial_embeddings)

        print("y_pred[-1]==hn: ", y[-1]==hn)

        return self.pred(y)
