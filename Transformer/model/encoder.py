import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from ..transformer_layers import *


class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, drop_prob=0.1):
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention
        self.mh_att = MultiHeadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.layernorm1 = LayerNorm(embed_dim)

        # Point-Wise Feed Forward
        self.ff = PointWiseFeedForward(embed_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layernorm2 = LayerNorm(embed_dim)
    
    def forward(self, src, src_mask):
        mh_out = self.mh_att(query=src, key=src, value=src, attn_mask=src_mask)
        mh_out = self.dropout1(mh_out)
        mh_out = self.layernorm1(src + mh_out)
        
        ff_out = self.ff(mh_out)
        ff_out = self.dropout2(ff_out)
        ff_out = self.layernorm2(mh_out + ff_out)

        return ff_out


class Encoder(nn.Module):

    def __init__(self, embed_dim, vocab_size, hidden_dim, num_heads, N, device):
        super(Encoder, self).__init__()

        self.tok_emb = TokenEmbedding(embed_dim, vocab_size)
        self.pos_enc = PositionalEncoding(embed_dim, device)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, hidden_dim, num_heads) for _ in range(N)])

    def forward(self, src, src_mask):
        src = self.tok_emb(src)
        src = self.pos_enc(src)

        for layer in self.encoder_blocks:
            src = layer(src, src_mask)
        
        return src


