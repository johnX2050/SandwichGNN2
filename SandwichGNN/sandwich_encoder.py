import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import BatchNorm2d, Conv1d, Conv2d,  LayerNorm, BatchNorm1d
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from SandwichGNN.attention import FullAttention, AttentionLayer, PositionwiseFeedForward
from SandwichGNN.t_components import RegularMask, Bottleneck_Construct, refer_points, \
    get_mask, PositionwiseFeedForward, MLP
from SandwichGNN.s_components import GNN
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from SandwichGNN.mtgnn_layer import *

class Encoderlayer(nn.Module):
    def __init__(self, next_n_nodes, layer_idx, num_nodes=0, gcn_true=True, buildA_true=True, gcn_depth=2,  device='cuda:0', predefined_A=None,
                 static_feat=None, dropout=0.3, skip_channels=64,
                 subgraph_size=40, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32,
                 seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05,
                 tanhalpha=3, layer_norm_affline=True, normalize_before=None, assign='rand'):
        super(Encoderlayer, self).__init__()
        self.next_n_nodes = next_n_nodes
        self.layer_idx = layer_idx
        self.n_layers = layers
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.assign = assign
        self.residual_channels = residual_channels
        self.filter_convs_1 = nn.ModuleList()
        self.filter_convs_2 = nn.ModuleList()
        self.gate_convs_1 = nn.ModuleList()
        self.gate_convs_2 = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.gconv4 = nn.ModuleList()
        self.gconv5 = nn.ModuleList()
        self.gconv6 = nn.ModuleList()
        self.gconv7 = nn.ModuleList()
        self.gconv8 = nn.ModuleList()
        self.norm = nn.ModuleList()

        # for me
        self.time_conv1s = nn.ModuleList()
        self.time_conv2s = nn.ModuleList()
        self.filter_conv1s = nn.ModuleList()
        self.filter_conv2s = nn.ModuleList()
        self.skip_conv1s = nn.ModuleList()
        self.skip_conv2s = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()
        self.norm3s = nn.ModuleList()
        self.norm4s = nn.ModuleList()
        self.mlp_after_s1s = nn.ModuleList()
        self.mlp_after_s2s = nn.ModuleList()
        self.self_attns = nn.ModuleList()
        self.pos_ffns = nn.ModuleList()


        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length

        kernel_size = 7
        self.kernel_size_ = kernel_size
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

                self.filter_convs_1.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.filter_convs_2.append(dilated_inception_small_seq(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs_1.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs_2.append(dilated_inception_small_seq(residual_channels, conv_channels, dilation_factor=new_dilation))
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
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv3.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv4.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv5.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv6.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv7.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv8.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

                # for me
                self.time_conv1s.append(Conv2d(residual_channels, 2*residual_channels, kernel_size=(1, 5), padding=(0, 2),
                                              stride=(1, 1), bias=True, dilation=1))
                self.time_conv2s.append(Conv2d(residual_channels, 2 * residual_channels, kernel_size=(1, 3), padding=(0, 1),
                                        stride=(1, 1), bias=True, dilation=1))
                self.filter_conv1s.append(Conv2d(residual_channels, residual_channels, kernel_size=(1, 12), padding=(0, 0),
                                           stride=(1, 1), bias=True, dilation=1))
                self.filter_conv2s.append(Conv2d(residual_channels, residual_channels, kernel_size=(1, 6), padding=(0, 0),
                                           stride=(1, 1), bias=True, dilation=1))
                self.skip_conv1s.append(nn.Conv2d(in_channels=conv_channels,
                                            out_channels=skip_channels,
                                            kernel_size=(1, 12)))
                self.skip_conv2s.append(nn.Conv2d(in_channels=conv_channels,
                                            out_channels=skip_channels,
                                            kernel_size=(1, 6)))

                self.norm1s.append(LayerNorm((conv_channels, num_nodes, 12), elementwise_affine=layer_norm_affline))
                self.norm2s.append(LayerNorm((conv_channels, num_nodes, 12), elementwise_affine=layer_norm_affline))
                self.norm3s.append(LayerNorm((conv_channels, num_nodes, 6), elementwise_affine=layer_norm_affline))
                self.norm4s.append(LayerNorm((conv_channels, num_nodes, 6), elementwise_affine=layer_norm_affline))

                self.mlp_after_s1s.append(torch.nn.Conv2d(2 * residual_channels, residual_channels,
                                                    kernel_size=(1, 3), padding=(0, 1), stride=(1, 1), bias=True))
                self.mlp_after_s2s.append(torch.nn.Conv2d(2 * residual_channels, residual_channels,
                                                    kernel_size=(1, 3), padding=(0, 1), stride=(1, 1), bias=True))

                # for t attention
                d_x_in = 32
                self.window_size = [2]
                self.inner_size = 5
                self.in_len = seq_length
                d_model = 32
                n_heads = 4
                d_inner = 32

                self.self_attns.append(AttentionLayer(
                    FullAttention(mask_flag=True, factor=0,
                                  attention_dropout=dropout, output_attention=False),
                    d_model, n_heads
                ))
                self.pos_ffns.append(PositionwiseFeedForward(
                    d_model, d_inner, dropout=dropout, normalize_before=normalize_before))

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.layers = layers

        self.idx = torch.arange(self.num_nodes).to(device)

        # self.gnn1_pool = GNN(residual_channels, residual_channels, self.next_n_nodes)
        if assign == 'rand':
            self.assign_matrix = Parameter(torch.randn(num_nodes, next_n_nodes).to(device, non_blocking=True), requires_grad=True)
        elif assign == 'spectral_cluster':
            self.assign_matrix = np.float32(np.load('./data/assignment_matrix.npy'))
            self.assign_matrix = torch.tensor(self.assign_matrix).to(device)

        self.t_len_proj_mlp = Seq(Lin(int(self.seq_length * 1.5), self.seq_length), ReLU(inplace=True))

        # only construct the pyramid for the first layer
        self.d_bottleneck = d_x_in // 4
        self.conv_layers = Bottleneck_Construct(
            d_x_in, self.window_size, self.d_bottleneck)
        #
        # self.norm3 = torch.nn.LayerNorm([conv_channels, num_nodes, 1])
        # self.norm4 = torch.nn.LayerNorm([conv_channels, num_nodes, 1])

        self.skipEnd = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 12), bias=True)


    def forward(self, x, skip, idx=None):

        seq_len = x.size(3)
        # next_x_seq_len = seq_len - self.n_layers * self.kernel_size_
        skip = skip

        # assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        # if seq_len < self.receptive_field:
        #     x = nn.functional.pad(x,(self.receptive_field-seq_len,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        # construct pyramid
        x = rearrange(x, 'b c n l -> (b n) l c')
        x = self.conv_layers(x)

        # get attention mask
        b_n = x.shape[0]
        mask, all_size = get_mask(
            self.in_len, self.window_size, self.inner_size)
        self.indexes = refer_points(all_size, self.window_size)
        slf_attn_mask = mask.repeat(b_n, 1, 1).to(x.device)
        attn_mask = RegularMask(slf_attn_mask)

        x = rearrange(x, '(b n) l c -> b c n l', n=self.num_nodes)

        for i in range(self.layers):

            residual = x

            # spilt x to two parts
            x_fi = x[:, :, :, :seq_len]
            x_co = x[:, :, :, seq_len:]

            # T for the first part
            # x_fi = self.time_conv1s[i](x_fi)
            # x1, x2 = torch.split(x_fi, [self.residual_channels, self.residual_channels], 1)
            # x_fi = torch.tanh(x1) * torch.sigmoid(x2)
            # x_fi = F.dropout(x_fi, self.dropout, training=self.training)

            x_fi = nn.functional.pad(x_fi, (6, 0, 0, 0))
            filter = self.filter_convs_1[i](x_fi)
            filter = torch.tanh(filter)
            gate = self.gate_convs_1[i](x_fi)
            gate = torch.sigmoid(gate)
            x_fi = filter * gate
            x_fi = F.dropout(x_fi, self.dropout, training=self.training)

            # T for the second part
            # x_co = self.time_conv2s[i](x_co)
            # x1, x2 = torch.split(x_co, [self.residual_channels, self.residual_channels], 1)
            # x_co = torch.tanh(x1) * torch.sigmoid(x2)
            # x_co = F.dropout(x_co, self.dropout, training=self.training)

            x_co = nn.functional.pad(x_co, (2, 0, 0, 0))
            filter = self.filter_convs_2[i](x_co)
            filter = torch.tanh(filter)
            gate = self.gate_convs_2[i](x_co)
            gate = torch.sigmoid(gate)
            x_co = filter * gate
            x_co = F.dropout(x_co, self.dropout, training=self.training)

            # change the skip kernel size
            s1 = x_fi
            s2 = x_co
            s1 = self.skip_conv1s[i](s1)
            s2 = self.skip_conv2s[i](s2)
            skip = s1 + s2 + skip

            # # do conv to compress the channel to 1
            node_vec_fi = self.filter_conv1s[i](x_fi)
            # node_vec_fi = self.norm3(node_vec_fi)
            node_vec_fi = rearrange(node_vec_fi, 'b c n l -> b (n l) c')
            # not reasonable
            node_vec_fi = node_vec_fi.mean(dim=0)

            node_vec_co = self.filter_conv2s[i](x_co)
            # node_vec_co = self.norm4(node_vec_co)
            node_vec_co = rearrange(node_vec_co, 'b c n l -> b (n l) c')
            # not reasonable
            node_vec_co = node_vec_co.mean(dim=0)

            # generate the adj by multiplying the node embeddings
            # small scale
            A = F.relu(torch.mm(node_vec_fi, node_vec_fi.transpose(1, 0)))
            # d = 1 / (torch.sum(A, -1))
            # D = torch.diag_embed(d)
            # A = torch.matmul(D, A)

            # large scale
            A_large_scale = F.relu(torch.mm(node_vec_co, node_vec_co.transpose(1, 0)))
            # d_c = 1 / (torch.sum(A_large_scale, -1))
            # D_c = torch.diag_embed(d_c)
            # A_large_scale = torch.matmul(D_c, A_large_scale)

            # S
            if self.gcn_true:
                x_fi1 = self.gconv1[i](x_fi, adp)+self.gconv2[i](x_fi, adp.transpose(1,0))
                x_fi2 = self.gconv3[i](x_fi, A) + self.gconv4[i](x_fi, A.transpose(1, 0))
                if idx is None:
                    x_fi1 = self.norm1s[i](x_fi1, self.idx)
                    x_fi2 = self.norm2s[i](x_fi2, self.idx)
                else:
                    x_fi1 = self.norm1s[i](x_fi1, idx)
                    x_fi2 = self.norm2s[i](x_fi2, self.idx)
                x_fi = torch.cat([x_fi1, x_fi2], dim=1)
                x_fi = self.mlp_after_s1s[i](x_fi)

                x_co1 = self.gconv5[i](x_co, adp)+self.gconv6[i](x_co, adp.transpose(1,0))
                x_co2 = self.gconv7[i](x_co, A_large_scale)+self.gconv8[i](x_co, A_large_scale.transpose(1,0))
                if idx is None:
                    x_co1 = self.norm3s[i](x_co1, self.idx)
                    x_co2 = self.norm4s[i](x_co2, self.idx)
                else:
                    x_co1 = self.norm3s[i](x_co1, idx)
                    x_co2 = self.norm4s[i](x_co2, self.idx)
                x_co = torch.cat([x_co1, x_co2], dim=1)
                x_co = self.mlp_after_s2s[i](x_co)

            else:
                x_fi = self.residual_convs[i](x_fi)
                x_co = self.residual_convs[i](x_co)

            # concat the small scale and large scale
            x_after_ts = torch.cat([x_fi, x_co], dim=-1)
            x_after_ts = x_after_ts + residual

            x = rearrange(x, 'b c n l -> (b n) l c')
            x_after_ts = rearrange(x_after_ts, 'b c n l -> (b n) l c')

            # do the pyramid attention
            output, _ = self.self_attns[i](
                x_after_ts, x_after_ts, x_after_ts, attn_mask=attn_mask
            )
            x = self.pos_ffns[i](output)

            x = rearrange(x, '(b n) l c -> b c n l', n=self.num_nodes)

        x = self.t_len_proj_mlp(x)

        # if: current block is the last, ouput next x
        # otherwise: no need to output next x, since it's at the top encoder layer
        if self.layer_idx != 2:
            # Get next x, follow GAGNN
            ass = self.assign_matrix
            # ass = repeat(ass, 'n_nodes next_n_nodes -> b n_nodes next_n_nodes', b=batch_size)
            ass = rearrange(ass, 'n m->m n')
            # need to be corrected
            next_x = torch.einsum("bcnt, mn->bcmt", [x, ass])

        # do the skip conv
        skip = self.skipEnd(x) + skip
        x_out = F.relu(skip)

        # if cur block is the last encoder block, just output the x, adj and s
        # otherwise: pass the 5 variables
        # 2 is dynamic by the num of blocks
        if self.layer_idx != 2:
            return x_out, adp, ass, next_x, skip
        else:
            return x_out, adp



class Encoder(nn.Module):
    """
    Description: The Encoder compose of encoder layers.
    Input: x
    Output: encoder_layer_outputs, adp, s
    """

    def __init__(self, seq_len, n_nodes=207, n_layers=2, in_dim=8, residual_channels=32,
                 skip_channels=64,
                 s_factor=20, predefined_A=None,
                 dropout=0.3
                 ):
        super(Encoder, self).__init__()

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.s_factor = s_factor
        self.predefined_A = predefined_A
        self.dropout = dropout

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.skip0 = nn.Conv2d(in_channels=residual_channels,
                               out_channels=skip_channels,
                               kernel_size=(1, self.seq_len))

        self.encoder_layers = nn.ModuleList([
            Encoderlayer(num_nodes=207, layer_idx=1, next_n_nodes=int(n_nodes // s_factor), predefined_A=self.predefined_A,
                         subgraph_size=40,
                         seq_length=self.seq_len, assign='rand')]
        )

        for i in range(1, n_layers):
            cur_n_nodes = int(n_nodes // math.pow(s_factor, i))
            next_n_nodes = int(cur_n_nodes // s_factor)
            # seq_length = self.seq_len - (18 * i)
            self.encoder_layers.append(
                Encoderlayer(num_nodes=cur_n_nodes, layer_idx=i+1, next_n_nodes=next_n_nodes,
                             subgraph_size=5,
                             seq_length=self.seq_len, assign='rand')
            )

    def forward(self, x, idx=None):
        enc_outputs = []
        enc_adp = []
        enc_s = []
        encoder_outputs = []

        x = self.start_conv(x)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        # encoder layer 0
        enc_out, adp, s, next_enc_in, skip = self.encoder_layers[0](x, skip, idx)
        # # transpose s
        # s = rearrange(s, 'n m->m n')
        # get next skip
        next_skip = torch.einsum("bcnt, mn -> bcmt", [skip, s])
        # get original skip
        skip_ori = skip

        # append encoder layer 1 outputs: x, adp and s
        enc_outputs.append(enc_out)
        enc_adp.append(adp)
        enc_s.append(s)

        for i in range(1, self.n_layers):
            if i != self.n_layers-1:
                enc_out, adp, s, next_enc_in, skip = self.encoder_layers[i](next_enc_in, next_skip, None)
                next_skip = torch.einsum("bcnt, mn -> bcmt", [skip, s])
                enc_s.append(s)
            else:
                enc_out, adp = self.encoder_layers[i](next_enc_in, next_skip, None)
            # # transpose s
            # s = rearrange(s, 'n m-> m n')
            # get next skip


            enc_outputs.append(enc_out)
            enc_adp.append(adp)

        encoder_outputs.append(enc_outputs)
        encoder_outputs.append(enc_adp)
        encoder_outputs.append(enc_s)

        return encoder_outputs, skip_ori