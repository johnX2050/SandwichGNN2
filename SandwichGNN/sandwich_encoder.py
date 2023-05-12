import math
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from SandwichGNN.attention import FullAttention, AttentionLayer, PositionwiseFeedForward
from SandwichGNN.t_components import RegularMask, Bottleneck_Construct, refer_points, \
    get_mask, PositionwiseFeedForward, MLP
from SandwichGNN.s_components import GNN
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from SandwichGNN.mtgnn_layer import *

class Encoderlayer(nn.Module):
    def __init__(self, next_n_nodes, num_nodes=0, gcn_true=True, buildA_true=True, gcn_depth=2,  device='cuda:0', predefined_A=None,
                 static_feat=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32,
                 seq_length=72, in_dim=2, out_dim=12, layers=2, propalpha=0.05,
                 tanhalpha=3, layer_norm_affline=True):
        super(Encoderlayer, self).__init__()
        self.next_n_nodes = next_n_nodes
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 13
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

                self.filter_convs.append(dilated_inception_long_seq(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception_long_seq(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers

        self.idx = torch.arange(self.num_nodes).to(device)

        # self.gnn1_pool = GNN(residual_channels, residual_channels, self.next_n_nodes)
        self.assign_matrix = Parameter(torch.randn(num_nodes, next_n_nodes).to(device, non_blocking=True), requires_grad=True)


    def forward(self, x, idx=None):

        batch_size = x.size(0)
        seq_len = x.size(3)

        # assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if seq_len < self.receptive_field:
            x = nn.functional.pad(x,(self.receptive_field-seq_len,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        for i in range(self.layers):
            residual = x

            # T
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            # S
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x_out = self.norm[i](x,self.idx)
            else:
                x_out = self.norm[i](x,idx)

        ass = self.assign_matrix
        ass = repeat(ass, 'n_nodes next_n_nodes -> b n_nodes next_n_nodes', b=batch_size)
        ass = rearrange(ass, 'b n m->b m n')
        # need to be corrected
        next_x_in = torch.bmm(ass, x_out)


        # get the assignment matrix and more coarsen nodes embeddings
        # this section can be replaced by randomly initializing s
        # assignment_matrix = self.gnn1_pool(x, adp)
        # next_x, next_adp, _, _ = dense_diff_pool(x, adp, assignment_matrix)

        return x_out, adp, self.assign_matrix, next_x_in




class Encoder(nn.Module):
    """
    Description: The Encoder compose of encoder layers.
    Input: x
    Output: encoder_layer_outputs, adp, s
    """

    def __init__(self, n_nodes=207, n_layers=3, seq_len=None, in_dim=2, residual_channels=32,
                s_factor=4, predefined_A=None
                 ):
        super(Encoder, self).__init__()

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.s_factor = s_factor
        self.predefined_A = predefined_A

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.encoder_layers = nn.ModuleList([
            Encoderlayer(num_nodes=207, next_n_nodes=int(n_nodes // s_factor), predefined_A=self.predefined_A)]
        )

        for i in range(1, n_layers):
            cur_n_nodes = int(n_nodes // math.pow(s_factor, i))
            next_n_nodes = int(cur_n_nodes // s_factor)
            self.encoder_layers.append(
                Encoderlayer(num_nodes=cur_n_nodes, next_n_nodes=next_n_nodes)
            )

    def forward(self, x):
        enc_outputs = []
        enc_adp = []
        enc_s = []
        encoder_outputs = []


        x = self.start_conv(x)

        # encoder layer 1
        enc_out, adp, s, next_enc_in = self.encoder_layers[0](x)

        # append encoder layer 1 outputs: x, adp and s
        enc_outputs.append(enc_out)
        enc_adp.append(adp)
        enc_s.append(s)


        for i in range(1, len(self.encode_blocks)):
            enc_out, adp, s, next_enc_in = self.encoder_layers[i](next_enc_in)

            enc_outputs.append(enc_out)
            enc_adp.append(adp)
            enc_s.append(s)

        encoder_outputs.append(enc_outputs)
        encoder_outputs.append(enc_adp)
        encoder_outputs.append(enc_s)

        return encoder_outputs