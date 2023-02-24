import torch
from torch.nn import Sequential, Linear, ReLU, SELU, LSTM
from torch_geometric.nn import NNConv


class GNN(torch.nn.Module):
    def __init__(self, edge_linear_in, edge_linear_out, conv_in, conv_out, hidden_layer_size, num_rnn_layers=1):
        super().__init__()

        nn = Sequential(Linear(edge_linear_in, edge_linear_out), ReLU())
        self.conv = NNConv(conv_in, conv_out, nn, aggr='mean')

        self.act = SELU()

        self.hidden_layer_size = hidden_layer_size
        self.rnn = LSTM(conv_out, hidden_layer_size, num_rnn_layers, batch_first=True)

    def forward(self, data, y):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        n_samples = y.size(0)

        x = self.act(self.conv(x, edge_index, edge_attr)) # x dim -> [n, d]
        out = torch.mean(x, dim=0, keepdim=True)

        h_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layer_size, dtype=torch.float32)
        y_pred, (hn, cn) = self.rnn(x)

        y_pred = x

        return y_pred
