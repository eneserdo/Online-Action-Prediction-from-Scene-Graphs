from typing import List
import torch
from torch.nn import Sequential, Dropout, Linear, ReLU, SELU, LSTM, Softmax, TransformerEncoder, \
    TransformerEncoderLayer, Module, functional as F
import torch_geometric.nn as geometric_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge
import math

from dataset import OBJECTS
from utils import get_filters_for_imbalance, align_embeddings_to_right


# Borrowed from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ActionRecognition(torch.nn.Module):
    def __init__(self, gnn_type,
                 temporal_type,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_edge_feat,
                 batch_size,
                 num_action_classes,
                 lstm_hidden_size,
                 use_bn,
                 filter_it,
                 use_hand_pooling,
                 edge_dropout,
                 dropout,
                 use_cls_token,
                 **kwargs):
        """ Accepts kwargs for non-general params of gnns """
        super().__init__()

        self.batch_size = batch_size
        self.filter_it = filter_it
        self.use_hand_pooling = use_hand_pooling
        self.edge_dropout = edge_dropout
        self.temporal_type = temporal_type
        self.use_cls_token = use_cls_token

        # Sets Graph Network
        if gnn_type == "NNConv":
            nn1 = Sequential(Linear(num_edge_feat, in_channels * hidden_channels), ReLU())
            self.conv1 = geometric_nn.NNConv(in_channels, hidden_channels, nn1, aggr='mean')
            nn2 = Sequential(Linear(num_edge_feat, hidden_channels * out_channels), ReLU())
            self.conv2 = geometric_nn.NNConv(hidden_channels, out_channels, nn2, aggr='mean')
            effective_out_ch1 = hidden_channels
            effective_out_ch2 = out_channels

        elif gnn_type in ["GATv2Conv", "TransformerConv", "GATConv"]:
            self.conv1 = getattr(geometric_nn, gnn_type)(in_channels, hidden_channels, heads=kwargs["heads"],
                                                         edge_dim=num_edge_feat, dropout=dropout)
            self.conv2 = getattr(geometric_nn, gnn_type)(hidden_channels * kwargs["heads"], out_channels,
                                                         heads=kwargs["heads"], edge_dim=num_edge_feat, dropout=dropout)
            effective_out_ch1 = hidden_channels * kwargs["heads"]
            effective_out_ch2 = out_channels * kwargs["heads"]

        else:
            raise NotImplementedError(f"gnn_type: {gnn_type} is not implemented")

        self.act1 = SELU()
        self.act2 = SELU()

        if use_bn:
            self.bn1 = geometric_nn.BatchNorm(effective_out_ch1)
            self.bn2 = geometric_nn.BatchNorm(effective_out_ch2)
            self.gnn = geometric_nn.Sequential("x, edge_index, edge_attr",
                                               [(self.conv1, "x, edge_index, edge_attr -> x"), self.act1, self.bn1,
                                                (self.conv2, "x, edge_index, edge_attr -> x"), self.act2, self.bn2])
        else:
            self.gnn = geometric_nn.Sequential("x, edge_index, edge_attr",
                                               [(self.conv1, "x, edge_index, edge_attr -> x"), self.act1,
                                                (self.conv2, "x, edge_index, edge_attr -> x"), self.act2])

        if temporal_type in ["lstm", "bi"]:
            num_rnn_layers = 1  # Lets make it constant
            is_bidirectional = ("bi" == temporal_type)

            # To have approx same number of parameters
            lstm_hidden_size = lstm_hidden_size * 2 // 3 if is_bidirectional else lstm_hidden_size

            self.temporal_layer = LSTM(effective_out_ch2, lstm_hidden_size, num_rnn_layers,
                                       bidirectional=is_bidirectional, dropout=dropout)
            lstm_hidden_size = lstm_hidden_size * 2 if is_bidirectional else lstm_hidden_size

        elif temporal_type == "tr":
            # Default dropout is 0.1
            self.temporal_layer = TransformerEncoder(
                TransformerEncoderLayer(effective_out_ch2, 4, lstm_hidden_size // 4 * 3), 2)
            self.PE_coef = math.sqrt(effective_out_ch2)
            self.pos_encoder = PositionalEncoding(effective_out_ch2)

            if use_cls_token:
                self.cls_tokens = torch.nn.Parameter(torch.randn(1, 1, effective_out_ch2)).expand(self.batch_size, -1,
                                                                                                  -1)

            lstm_hidden_size = effective_out_ch2

        self.mlp = Sequential(Linear(lstm_hidden_size, num_action_classes), Softmax(dim=-1))

        self.segment_boundary_detector = Sequential(Linear(lstm_hidden_size + effective_out_ch2, 2))

    def _hands_pooling(self, x, embeddings, batch):

        labels = torch.argmax(x, dim=1)

        left_hand_indicies = torch.zeros((self.batch_size), dtype=torch.long)
        right_hand_indicies = torch.zeros((self.batch_size), dtype=torch.long)

        for i, l in enumerate(labels):
            if l == OBJECTS.index("LeftHand"):
                left_hand_indicies[batch[i]] = i

            elif l == OBJECTS.index("RightHand"):
                right_hand_indicies[batch[i]] = i

        # left, right: [batch_size, len(embedding)]
        # return embeddings[graph_hand_indicies]
        return embeddings[left_hand_indicies], embeddings[right_hand_indicies]
    
    def _segment_prediction(self, gnn_output, rnn_output):
        hn_se = torch.concat([rnn_output, gnn_output], dim=-1)
        segment_change_pred = F.gumbel_softmax(self.segment_boundary_detector(hn_se), dim=-1)[:,:,:1]
        s_hard = (segment_change_pred > 0.5).float()
        hard_segment_change = ((s_hard - segment_change_pred).detach() + segment_change_pred).squeeze()
        return segment_change_pred, hard_segment_change.to(torch.long)
        

    def forward(self, dataBatchList: List[Batch]):

        left_embedding_list = []
        right_embedding_list = []

        for databatch in dataBatchList:  # processes each frame time-independently
            x, edge_attr, edge_index, batch = databatch.x, databatch.edge_attr, databatch.edge_index, databatch.batch

            if self.edge_dropout:
                edge_index, edge_mask = dropout_edge(edge_index, p=0.1)
                edge_attr = edge_attr[edge_mask]

            H = self.gnn(x, edge_index, edge_attr)

            if self.use_hand_pooling:
                left_embeddings, right_embeddings = self._hands_pooling(x, H, batch)
            else:
                # A bit inefficient implementation
                graph_embedding = geometric_nn.pool.global_mean_pool(H, batch)
                left_embeddings = graph_embedding
                right_embeddings = graph_embedding

            left_embedding_list.append(left_embeddings)
            right_embedding_list.append(right_embeddings)

            # assert graph_embedding.shape[0]==self.batch_size and graph_embedding.shape[1]==self.conv_out

        left_sequencial_embeddings = torch.stack(left_embedding_list)
        right_sequencial_embeddings = torch.stack(right_embedding_list)

        # Filtering imbalanced data
        if self.filter_it and self.training:
            # get filters
            left_filter, right_filter = get_filters_for_imbalance(dataBatchList)

            # filter it
            right_sequencial_embeddings[~right_filter] = 0
            left_sequencial_embeddings[~left_filter] = 0

            # align non-zero embeddings to the right
            # TODO: Insert assert statement to make sure it is working as expectedly            
            right_sequencial_embeddings = align_embeddings_to_right(right_sequencial_embeddings, right_filter)
            left_sequencial_embeddings = align_embeddings_to_right(left_sequencial_embeddings, left_filter)

        both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=1)

        if self.temporal_type == "tr":
            if self.use_cls_token:
                both_sequencial_embeddings = torch.cat((self.cls_tokens, both_sequencial_embeddings), dim=1)

            both_sequencial_embeddings = self.pos_encoder(both_sequencial_embeddings * self.PE_coef)

            both_hn = self.temporal_layer(both_sequencial_embeddings)

            if self.use_cls_token:
                both_hn = both_hn[0]
            else:
                both_hn = both_hn.mean(dim=0)  # FIXME: It is dim=1 in the original code ?


        elif self.temporal_type == "bi":
            both_out, _ = self.temporal_layer(both_sequencial_embeddings)
            """
            _, (both_hn, _) = self.temporal_layer(both_sequencial_embeddings)
            both_hn = torch.cat([both_hn[0], both_hn[1]], dim=1)
            """
        else:
            both_out, _ = self.temporal_layer(both_sequencial_embeddings)
            
            """_, (both_hn, _) = self.temporal_layer(both_sequencial_embeddings)
            # assert both_hn.shape[0]==1
            both_hn = both_hn.squeeze()"""

        # Segmentation task - predict segment change (0,1)
        segment_change_pred, hard_segment_change_pred = self._segment_prediction(both_sequencial_embeddings, both_out)
        

        # mask 
        
        both_out[hard_segment_change_pred]


        # Label prediction task
        action_prediction = self.mlp(both_out) 

        return action_prediction, segment_change_pred
