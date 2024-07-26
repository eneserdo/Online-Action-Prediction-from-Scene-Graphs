"""
Notes:

Shape abbreviations:
    T: temporal dimension
    N: batch dimension
    F: feature dimension
"""

import math
import warnings
from typing import List

import torch
import torch.nn as nn
import torch_geometric.nn as geometric_nn
from positional_encodings.torch_encodings import (PositionalEncoding1D,
                                                  PositionalEncoding2D)
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge

from dataset import ACTIONS, OBJECTS, RELATIONS
from utils import (align_embeddings_to_right, flatten_seq,
                   get_filters_for_imbalance, print_warning)


# Borrowed from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
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

    def __init__(self, d_model, dropout=0, max_len=5000, use_library=False):
        super(PositionalEncoding, self).__init__()
        # FIXME: dropout ??
        self.dropout = nn.Dropout(p=dropout)

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
    
class PositionalEncodingPseudo2D(nn.Module):
    """
    Same as previous but with pseudo 2D positional encoding. Instead of (T, 2), it is ([T1, T2])
    """

    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncodingPseudo2D, self).__init__()
        # FIXME: dropout ??
        self.dropout = nn.Dropout(p=dropout)
        self.pe_ = PositionalEncoding2D(d_model)
        self.is_set = False
        
    def _set_pe(self, shape, device):
        
        x_dim, bs, d_model = shape

        self.x_dim = x_dim

        y_dim = 2
        self.zeros = torch.zeros((bs, x_dim, y_dim, d_model), dtype=torch.float32).to(device)
        
        self.pe_obtained = self.pe_(self.zeros)

        self.pe_y0 = self.pe_obtained[:, :, 0, :].transpose(0, 1)

        self.pe_y1 = self.pe_obtained[:, :, 1, :].transpose(0, 1)

        self.pe = torch.cat([self.pe_y0, self.pe_y1], dim=1)
        self.is_set = True


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

        if not self.is_set:
            self._set_pe(x.shape, x.device)

        assert x.shape == self.x_dim.shape

        x = x + self.pe

        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    """

    def __init__(self, seg_len, d_model, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Parameter(torch.zeros(seg_len, 1, d_model), requires_grad=True)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


# Borrowed from https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        # self.generator = nn.nn.Linear(emb_size, tgt_vocab_size)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None):
        
        embedding_size = src.shape[-1]
        scale = math.sqrt(embedding_size)
        src_emb = self.positional_encoding(src*scale)
        tgt_emb = self.positional_encoding(trg*scale)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # return self.generator(outs)
        return outs

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(src), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(tgt), memory, tgt_mask)


class GraphModule(nn.Module):
    def __init__(self, *, gnn_type,
                          in_channels,
                          hidden_channels,
                          out_channels,
                          num_edge_feat,
                          num_heads: int,
                          num_layers: int,
                          filter_it,
                          use_hand_pooling,
                          edge_dropout,
                          dropout,
                          norm_layer,
                          use_pos,
                          use_embedding):
        super().__init__()
        self.filter_it = filter_it
        self.use_hand_pooling = use_hand_pooling
        self.edge_dropout = edge_dropout
        self.use_embedding = use_embedding
        self.use_pos = use_pos

        if use_embedding:
            self.embedding = torch.nn.Embedding(in_channels, in_channels)


        assert norm_layer != "none", "norm_layer should not be none"
        assert norm_layer != "PairNorm", "PairNorm is not implemented yet"
        assert num_layers >= 2, "num_layers should be at least 2"

        NormLayerType = getattr(geometric_nn, norm_layer)

        if norm_layer == "BatchNorm":
            bn_transfer_string = "x -> x"
        else:
            bn_transfer_string = "x, batch -> x"

        hc_nh = hidden_channels*num_heads

        module_list = []
        # First layer
        module_list.extend([(getattr(geometric_nn, gnn_type)(in_channels, hidden_channels, heads=num_heads, edge_dim=num_edge_feat, dropout=dropout), "x, edge_index, edge_attr -> x"), 
                            nn.SELU(), 
                           (NormLayerType(hc_nh), bn_transfer_string)])


        # Middle layers
        for _ in range(num_layers-2):
            module_list.extend([(getattr(geometric_nn, gnn_type)(hc_nh, hidden_channels, heads=num_heads, edge_dim=num_edge_feat, dropout=dropout), "x, edge_index, edge_attr -> x"), 
                               nn.SELU(), 
                               (NormLayerType(hc_nh), bn_transfer_string)])
        
        # Last layer
        module_list.extend([(getattr(geometric_nn, gnn_type)(hc_nh, out_channels, heads=num_heads, edge_dim=num_edge_feat, dropout=dropout), "x, edge_index, edge_attr -> x"), 
                            nn.SELU(), 
                           (NormLayerType(hc_nh), bn_transfer_string)])


        self.gnn = geometric_nn.Sequential("x, edge_index, edge_attr, batch", module_list)

    def _hands_pooling(self, x, embeddings):
        
        labels = torch.argmax(x, dim=1)

        # Below code is much more optimal, use when data redundancy is fixed.
        l_emb, r_emb = embeddings[labels == OBJECTS.index("LeftHand")], embeddings[labels == OBJECTS.index("RightHand")]
        assert l_emb.shape == r_emb.shape

        return l_emb, r_emb
   
    def forward(self, dataBatch: Batch):
    
        x, edge_attr, edge_index, batch = dataBatch.x, dataBatch.edge_attr, dataBatch.edge_index, dataBatch.batch

        if hasattr(dataBatch, "pos"):
            pos_3d = dataBatch.pos

        if self.edge_dropout != 0.0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout)
            edge_attr = edge_attr[edge_mask]

        if self.use_embedding:
            x_ = self.embedding(x.argmax(dim=1).long())
        else:
            x_ = x
        
        if self.use_pos:
            x_ = torch.cat([x_, pos_3d], dim=-1)

        H = self.gnn(x_, edge_index, edge_attr, batch)

        if self.use_hand_pooling:
            left_embeddings, right_embeddings = self._hands_pooling(x, H)
        else:
            # A bit inefficient implementation
            graph_embedding = geometric_nn.pool.global_mean_pool(H, batch)
            left_embeddings = graph_embedding
            right_embeddings = graph_embedding
        
        return left_embeddings, right_embeddings



class ActionRecognition(nn.Module):
    def __init__(self, *, gnn_type,
                 temporal_type,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_edge_feat,
                 num_action_classes,
                 lstm_hidden_size,
                 num_heads,
                 num_layers,
                 filter_it,
                 use_hand_pooling,
                 edge_dropout,
                 dropout,
                 dropout_tr,
                 merged_pred,
                 norm_layer,
                 norm_first,
                 pos_enc,
                 temporal_length,
                 num_of_temp_layers,
                 dim_feedforward,
                 num_heads_tr,
                 use_pos,
                 use_embedding):
        """ Accepts kwargs for non-general params of gnns """
        super().__init__()

        self.filter_it = filter_it
        self.use_hand_pooling = use_hand_pooling
        self.edge_dropout = edge_dropout
        self.temporal_type = temporal_type
        self.merged_pred = merged_pred
        self.use_embedding = use_embedding

        
        self.gnn = GraphModule(gnn_type=gnn_type,
                               in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               out_channels=out_channels,
                               num_edge_feat=num_edge_feat,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               filter_it=filter_it,
                               use_hand_pooling=use_hand_pooling,
                               edge_dropout=edge_dropout,
                               dropout=dropout,
                               norm_layer=norm_layer,
                               use_pos=use_pos,
                               use_embedding=use_embedding)
        

        gnn_output_ch = out_channels * num_heads # aka effective_out_ch2

        if self.merged_pred == "early":
            gnn_output_ch *= 2

        if dim_feedforward == 0:
            # This is was the previous default value, so I am keeping it
            # But this makes the embedding size > dim_feedforward, which is not good
            dim_feedforward = lstm_hidden_size // 4 * 3

        if temporal_type in ["lstm", "bi"]:
            if num_of_temp_layers != 1:
                raise NotImplementedError("args.num_of_temp_layers is not 1 as used to be. Are you sure?")
            num_rnn_layers = 1  # Lets make it constant
            is_bidirectional = ("bi" == temporal_type)

            # To have approx same number of parameters
            lstm_hidden_size = lstm_hidden_size * 2 // 3 if is_bidirectional else lstm_hidden_size

            self.temporal_layer = nn.LSTM(gnn_output_ch, lstm_hidden_size, num_rnn_layers,
                                       bidirectional=is_bidirectional, dropout=dropout)
            lstm_hidden_size = lstm_hidden_size * 2 if is_bidirectional else lstm_hidden_size

        elif temporal_type == "tr":
            # Default dropout is 0.1

            self.temporal_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(gnn_output_ch, num_heads_tr, dim_feedforward, norm_first=norm_first, dropout=dropout_tr), num_of_temp_layers)
            self.PE_coef = math.sqrt(gnn_output_ch)

            if pos_enc == "original":
                # if merged_pred == "attention":
                #     self.pos_encoder = PositionalEncodingPseudo2D(effective_out_ch2)
                # else:
                #     self.pos_encoder = PositionalEncodingPseudo2D(effective_out_ch2)
                self.pos_encoder = PositionalEncoding(gnn_output_ch)
                # self.pos_encoder = PositionalEncoding1D(effective_out_ch2)

            elif pos_enc == "learnable":
                if merged_pred == "attention":
                    self.pos_encoder = LearnablePositionalEncoding(temporal_length*2, gnn_output_ch)
                else:    
                    self.pos_encoder = LearnablePositionalEncoding(temporal_length, gnn_output_ch)
                
                # raise NotImplementedError("Learnable positional encoding is not implemented yet")
            
            elif pos_enc == "none":
                self.pos_encoder = lambda x: x
            
            else:
                raise NotImplementedError(f"pos_enc: {pos_enc} is not implemented")
            
            # self.cls_tokens = torch.nn.Parameter(torch.randn(1, 1, effective_out_ch2)).expand(self.batch_size, -1, -1)

            lstm_hidden_size = gnn_output_ch

        elif temporal_type == "edtr":
            # Encoder-Decoder Transformer
            self.temporal_layer = Seq2SeqTransformer(num_of_temp_layers, num_of_temp_layers, gnn_output_ch, num_heads_tr, dim_feedforward, dropout_tr)
            self.decoder_mask = torch.nn.Transformer.generate_square_subsequent_mask(temporal_length)
        
        elif temporal_type == "none":
            # self.temporal_layer = torch.nn.Sequential(torch.nn.Linear(effective_out_ch2, effective_out_ch2), 
            #                                     torch.nn.SELU(),
            #                                     torch.nn.Linear(effective_out_ch2, num_action_classes))
            self.temporal_layer = torch.nn.Linear(gnn_output_ch, num_action_classes)

        def build_mlp(in_feat,out_feat, hid_feat=None):
            # hid_feat = (in_feat+out_feat)//2 if hid_feat == None else hid_feat
            # return Sequential(nn.Linear(in_feat, hid_feat), nn.SELU(), nn.Linear(hid_feat, out_feat))
            return nn.Linear(in_feat,out_feat)

        if self.temporal_type != "none":
            if self.merged_pred == "late":
                # Can this run in parallel? :smile:
                self.left_mlp = build_mlp(lstm_hidden_size*2, num_action_classes)
                self.right_mlp = build_mlp(lstm_hidden_size*2, num_action_classes)
                # self.segment_boundary_detector = build_mlp(lstm_hidden_size + effective_out_ch2, 2)
            elif self.merged_pred == "early":
                # self.segment_boundary_detector_L = build_mlp(lstm_hidden_size + effective_out_ch2, 2)
                # self.segment_boundary_detector_R = build_mlp(lstm_hidden_size + effective_out_ch2, 2)
                self.left_mlp = build_mlp(lstm_hidden_size, num_action_classes)
                self.right_mlp = build_mlp(lstm_hidden_size, num_action_classes)
                # self.mlp = build_mlp(lstm_hidden_size, num_action_classes)
            elif self.merged_pred in ["none", "attention"]:    
                self.mlp = build_mlp(lstm_hidden_size, num_action_classes)
                # self.segment_boundary_detector = build_mlp(lstm_hidden_size + effective_out_ch2, 2)



    def forward(self, dataBatchList: List[Batch], padding_mask=None):

        left_embedding_list = []
        right_embedding_list = []

        #### Graph part ####

        # processes each frame time-independently
        for databatch in dataBatchList:  

            left_embeddings, right_embeddings = self.gnn(databatch)

            left_embedding_list.append(left_embeddings)
            right_embedding_list.append(right_embeddings)


        left_sequencial_embeddings = torch.stack(left_embedding_list) # Shape: (T, N, F)
        right_sequencial_embeddings = torch.stack(right_embedding_list) # Shape: (T, N, F)

        if self.temporal_type != "none":
            # Filtering imbalanced data only during training
            if self.filter_it and self.training:
                raise NotImplementedError("Filtering may be broken, please check it")
                # get filters
                left_filter, right_filter = get_filters_for_imbalance(dataBatchList)

                # filter it
                right_sequencial_embeddings[~right_filter] = 0
                left_sequencial_embeddings[~left_filter] = 0

                # align non-zero embeddings to the right
                # TODO: Insert assert statement to make sure it is working as expectedly            
                right_sequencial_embeddings = align_embeddings_to_right(right_sequencial_embeddings, right_filter)
                left_sequencial_embeddings = align_embeddings_to_right(left_sequencial_embeddings, left_filter)

            # padding_mask: (N, T) 

            # Enabling the information flow between hands
            if self.merged_pred == "early": # concat in feature dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=-1)
                # shape = (T, N, F*2)
                if padding_mask is not None:
                    pass

            elif self.merged_pred == "attention": # concat in time dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=0)
                # shape = (T*2, N, F)
                # May not work due to weird mask
                if padding_mask is not None:
                    padding_mask = torch.concat([padding_mask, padding_mask], dim=1)

            elif self.merged_pred in ["late", "none"]: # concat in batch dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=1) 
                # shape = (T, N*2, F)
                if padding_mask is not None:
                    padding_mask = torch.concat([padding_mask, padding_mask], dim=0)
            else:
                raise NotImplementedError(f"merged_pred: {self.merged_pred} is not implemented")
            

            #### Temporal part ####
            if self.temporal_type == "tr":

                both_sequencial_embeddings = self.pos_encoder(both_sequencial_embeddings * self.PE_coef)

                both_out = self.temporal_layer(both_sequencial_embeddings, src_key_padding_mask=padding_mask)


            elif self.temporal_type == "bi":
                both_out, _ = self.temporal_layer(both_sequencial_embeddings)
                """_, (both_hn, _) = self.temporal_layer(both_sequencial_embeddings)
                both_hn = torch.cat([both_hn[0], both_hn[1]], dim=1)"""
            elif self.temporal_type == "lstm":
                both_out, _ = self.temporal_layer(both_sequencial_embeddings)
                """_, (both_hn, _) = self.temporal_layer(both_sequencial_embeddings)
                # assert both_hn.shape[0]==1
                both_hn = both_hn.squeeze()"""
            
            elif self.temporal_type == "edtr":
                # Encoder-Decoder Transformer, mask the decoder part 
                both_out = self.temporal_layer(both_sequencial_embeddings, both_sequencial_embeddings, None, self.decoder_mask)
                
            
            

            #### Prediction part ####
            if self.merged_pred == "late":
                # merged_out_s = self._merge_hands(out_s)
                # action_prediction = torch.cat((self.left_mlp(merged_out_s), self.right_mlp(merged_out_s)),dim=0)
                # both_out.shape = (T, N*2, F)
                # both_out_ml.shape = (T, N, F*2)
                
                both_out_left, both_out_right = torch.tensor_split(both_out, 2, dim=1) 

                both_out_ml = torch.cat([both_out_left, both_out_right], dim=-1)
                
                action_prediction = torch.cat((self.left_mlp(both_out_ml), self.right_mlp(both_out_ml)), dim=1)
                

            elif self.merged_pred == "early":
                raise NotImplementedError("Early merge is not implemented yet")
                # l_out_s, r_out_s = torch.tensor_split(out_s, 2, dim=0)
                # action_prediction = torch.cat((self.left_mlp(l_out_s), self.right_mlp(r_out_s)),dim=0)
                # OR
                #action_prediction = self.mlp(out_s)
            elif self.merged_pred == "attention":

                # both_out.shape = (T*2, N, F)

                both_out_left, both_out_right = torch.tensor_split(both_out, 2, dim=0)

                both_out = torch.cat([both_out_left, both_out_right], dim=1)

                action_prediction = self.mlp(both_out)

            elif self.merged_pred == "none":
                # both_out.shape = (T, N*2, F)
                action_prediction = self.mlp(both_out)
                
            else:
                raise NotImplementedError(f"merged_pred: {self.merged_pred} is not implemented")

        else:
            action_prediction_left = self.temporal_layer(left_sequencial_embeddings)
            action_prediction_right = self.temporal_layer(right_sequencial_embeddings)
            action_prediction = torch.cat((action_prediction_left, action_prediction_right), dim=1) 

        return action_prediction


class ActionRecognitionMonograph(nn.Module):
    def __init__(self, *, gnn_type,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_edge_feat,
                 num_action_classes,
                 num_heads,
                 num_layers,
                 filter_it,
                 edge_dropout,
                 dropout,
                 norm_layer,
                 use_pos,
                 use_embedding, 
                 temporal_length):
        """ Accepts kwargs for non-general params of gnns """
        super().__init__()

        self.filter_it = filter_it
        self.edge_dropout = edge_dropout
        self.use_embedding = use_embedding
        self.num_actions = num_action_classes
        self.temporal_length = temporal_length

        self.gnn = GraphModule(gnn_type=gnn_type,
                               in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               out_channels=out_channels,
                               num_edge_feat=num_edge_feat,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               filter_it=filter_it,
                               use_hand_pooling=True,
                               edge_dropout=edge_dropout,
                               dropout=dropout,
                               norm_layer=norm_layer,
                               use_pos=use_pos,
                               use_embedding=use_embedding)


        gnn_output_ch = out_channels * num_heads

        self.head = nn.Sequential(nn.Linear(gnn_output_ch, gnn_output_ch),
                                  nn.SELU(),
                                  nn.Linear(gnn_output_ch, num_action_classes))



    def _hands_pooling(self, x, embeddings):
        
        labels = torch.argmax(x, dim=1)

        # Below code is much more optimal, use when data redundancy is fixed.
        l_emb, r_emb = embeddings[labels == OBJECTS.index("LeftHand")], embeddings[labels == OBJECTS.index("RightHand")]
        assert l_emb.shape == r_emb.shape

        # left, right: [batch_size*K, len(embedding)]
        # K: the number of graphs in a monograph
        return l_emb, r_emb


    def forward(self, databatch: Batch, _=None):

        bs = databatch.batch[-1] + 1
        left_embeddings, right_embeddings = self.gnn(databatch) 
        # shape: (n*b, f)

        # assert graph_embedding.shape[0]==self.batch_size and graph_embedding.shape[1]==self.conv_out

        hands_embeddings = torch.cat((left_embeddings, right_embeddings), dim=0) # CHECKME
        
        action_prediction = self.head(hands_embeddings)

        left_actions, right_actions = torch.tensor_split(action_prediction, 2, dim=0)

        left_actions = left_actions.reshape(bs, self.temporal_length, self.num_actions).permute(1, 0, 2)
        right_actions = right_actions.reshape(bs, self.temporal_length, self.num_actions).permute(1, 0, 2)

        return torch.cat((left_actions, right_actions), dim=1)


def make_model(args):
    if args.use_pos:
        in_channels = len(OBJECTS) + 3
    elif args.use_vf:
        in_channels = 256
    else:
        in_channels = len(OBJECTS)

    if args.monograph:
        model = ActionRecognitionMonograph(gnn_type=args.gnn_type,
                                           in_channels=in_channels,
                                           hidden_channels=args.hidden_channels,
                                           out_channels=args.out_channels,
                                           num_edge_feat=len(RELATIONS) + 1,
                                           num_action_classes=len(ACTIONS),
                                           num_heads=args.num_heads,
                                           num_layers=args.num_layers,
                                           filter_it=args.filter_idle_hold,
                                           edge_dropout=args.edge_dropout,
                                           dropout=args.dropout,
                                           norm_layer=args.norm_layer,
                                           use_pos=args.use_pos,
                                           use_embedding=args.use_embedding,
                                           temporal_length=args.temporal_length)
    else:
        model = ActionRecognition(gnn_type=args.gnn_type,
                                        temporal_type=args.temporal_type,
                                        num_edge_feat=len(RELATIONS),
                                        in_channels=in_channels,
                                        hidden_channels=args.hidden_channels,
                                        out_channels=args.out_channels,
                                        num_action_classes=len(ACTIONS),
                                        lstm_hidden_size=args.lstm_hidden_size,
                                        num_heads=args.num_heads,
                                        num_layers=args.num_layers,
                                        filter_it=args.filter_idle_hold,
                                        use_hand_pooling=not args.use_global_pooling,
                                        edge_dropout=args.edge_dropout,
                                        dropout=args.dropout,
                                        dropout_tr=args.dropout_tr,
                                        merged_pred=args.merged_pred,
                                        norm_layer=args.norm_layer,
                                        norm_first=args.norm_first,
                                        pos_enc=args.pos_enc,
                                        temporal_length=args.temporal_length,
                                        num_of_temp_layers=args.num_of_temp_layers,
                                        dim_feedforward=args.dim_feedforward,
                                        num_heads_tr=args.num_heads_tr,
                                        use_pos=args.use_pos,
                                        use_embedding=args.use_embedding)
    return model