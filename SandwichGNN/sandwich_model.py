import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from SandwichGNN.sandwich_encoder import Encoder
from SandwichGNN.sandwich_decoder import Decoder
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from SandwichGNN.mtgnn_layer import *
from einops import rearrange

from math import ceil


class SandwichGNN(nn.Module):
    def __init__(self, d_model=32, seq_len=72, predefined_A=None):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super(SandwichGNN, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.predefined_A = predefined_A

        # Embedding
        # self.x_embed = nn.Linear(2, d_model)

        # Encoder
        self.encoder = Encoder(predefined_A=self.predefined_A, seq_len=72)

        # Decoder
        self.decoder = Decoder()

    def forward(self, x):

        seq_len = x.shape[3]
        assert seq_len == self.seq_len, 'input sequence length not equal to preset sequence length'

        # x = rearrange(x, 'b c n t -> b t n c')
        # x_embed = self.x_embed(x)
        # x_embed = rearrange(x_embed, 'b c n t -> b t n c')

        encoder_outputs, skip_ori = self.encoder(x)
        predict_y = self.decoder(encoder_outputs, skip_ori)
        predict_y = predict_y[:, :, :, 0]

        return predict_y