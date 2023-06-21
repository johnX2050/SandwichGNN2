import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from SandwichGNN.attention import FullAttention, AttentionLayer, PositionwiseFeedForward
from SandwichGNN.t_components import RegularMask, Bottleneck_Construct, refer_points, \
    get_mask, PositionwiseFeedForward, MLP
from SandwichGNN.s_components import GNN
from SandwichGNN.mtgnn_layer import *
from torch.nn.utils import weight_norm
# from torch_geometric.nn import GCNConv, norm
from math import ceil

class DecoderLayer(nn.Module):
    def __init__(self, gcn_true=True, gcn_depth=2, num_nodes=0, device='cuda:0', predefined_A=None,
                 dropout=0.3,
                 dilation_exponential=1, conv_channels=32, residual_channels=32,
                 skip_channels=64, end_channels=128, seq_length=12, in_dim=32, out_dim=12, layers=1, propalpha=0.05,
                 layer_norm_affline=True):
        super(DecoderLayer, self).__init__()

        self.gcn_true = gcn_true
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.num_nodes = num_nodes
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.idx = torch.arange(self.num_nodes).to(device)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                if self.gcn_true:
                    self.gconv1.append(mixprop(skip_channels, skip_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(skip_channels, skip_channels, gcn_depth, dropout, propalpha))

                # if self.seq_length>self.receptive_field:
                #     self.norm.append(LayerNorm((skip_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                # else:
                #     self.norm.append(LayerNorm((skip_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                self.norm.append(LayerNorm((skip_channels, num_nodes, self.seq_length),
                                           elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers

        # if self.seq_length > self.receptive_field:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
        #                            bias=True)
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
        #                            kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        #
        # else:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
        #                            kernel_size=(1, self.receptive_field), bias=True)
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
        #                            bias=True)


    def forward(self, x, adp, idx=None):

        # there is T and S in one decoder layer
        seq_len = x.shape[3]

        # padding the sequence if the seq length is not long enough
        # if seq_len < self.receptive_field:
        #     x = nn.functional.pad(x,(self.receptive_field-seq_len,0,0,0))

        # no skip connection

        for i in range(self.layers):
            residual = x

            # # without T
            # filter = self.filter_convs[i](x)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](x)
            # gate = torch.sigmoid(gate)
            # x = filter * gate
            # x = F.dropout(x, self.dropout, training=self.training)

            # S
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        return x




class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, n_layers=2, residual_channels=32, skip_channels=64,end_channels=128, out_dim=12):
        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.decoder_layers = nn.ModuleList()

        # declare 0 th decoder layer
        # the sequence length is not fixed
        # 10 is dynamic
        self.decoder_layers.append(DecoderLayer(num_nodes=10, seq_length=1))

        self.skipE = nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels,
                               kernel_size=(1,n_layers), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        # declare 1, 2... decoder layers
        for i in range(1, n_layers):
            seq_length = i + 1
            if i == 1:
                n_nodes = 207
            # elif i == 2:
            #     seq_length = 3
            #     n_nodes = 207
            self.decoder_layers.append(DecoderLayer(num_nodes=n_nodes, seq_length=seq_length))

    def forward(self, encoder_outputs, skip_ori):
        layers = self.decoder_layers

        # reverse all encoder outputs
        x = encoder_outputs[0]
        x.reverse()
        adj = encoder_outputs[1]
        adj.reverse()
        s = encoder_outputs[2]
        s.reverse()

        # the first decoder layer forward
        x_out = layers[0](x[0], adj[0])

        for i in range(1, self.n_layers):
            # transpose s[i]
            s[i-1] = rearrange(s[i-1], 'm n -> n m')
            next_x = torch.einsum("nm, bcmt->bcnt",[s[i-1], x_out])
            next_dec_in = torch.cat([x[i], next_x], dim=3)
            x_out = layers[i](next_dec_in, adj[i])

        # how to deal with x_out? (64, 32, 207, 61) -> (64, 12, 207, 1)
        # end cov can make 32 -> 12
        # This version use one skip conv to make 61 -> 1

        x = self.skipE(x_out) + skip_ori
        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        pred = self.end_conv_2(x)

        return pred