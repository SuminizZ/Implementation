import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from ..transformer_layers import *


class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, drop_prob=0.1):
        super(DecoderBlock, self).__init__()
        
        # Masked Multi-Head Attention
        self.masked_mh_att = MultiHeadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.layernorm1 = LayerNorm(embed_dim)

        # Multi-Head Attention
        self.mh_att = MultiHeadAttention(embed_dim, num_heads)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layernorm2 = LayerNorm(embed_dim)

        # Point-Wise Feed Forward
        self.ff = PointWiseFeedForward(embed_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.layernorm3 = LayerNorm(embed_dim)
    
    def forward(self, trg, src_enc, trg_mask, subsq_mask, src_trg_mask):
        masked_mh_out = self.masked_mh_att(query=trg, key=trg, value=trg, attn_mask=trg_mask, subsq_mask=subsq_mask)
        masked_mh_out = self.dropout1(masked_mh_out)
        masked_mh_out = self.layernorm1(trg + masked_mh_out)

        mh_out = self.mh_att(query=masked_mh_out, key=src_enc, value=src_enc, attn_mask=src_trg_mask)
        mh_out = self.dropout2(mh_out)
        mh_out = self.layernorm2(masked_mh_out + mh_out)
        
        ff_out = self.ff(mh_out)
        ff_out = self.dropout3(ff_out)
        ff_out = self.layernorm3(mh_out + ff_out)

        return ff_out


class Decoder(nn.Module):

    def __init__(self, embed_dim, vocab, hidden_dim, num_heads, N, device):
        super(Decoder, self).__init__()

        self.tok_emb = TokenEmbedding(embed_dim, vocab)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.encoder_blocks = nn.ModuleList([DecoderBlock(embed_dim, hidden_dim, num_heads) for _ in range(N)])

    def forward(self, trg, src_enc, trg_mask, subsq_mask, src_trg_mask):
        trg = self.tok_emb(trg)
        trg = self.pos_enc(trg)

        for layer in self.encoder_blocks:
            trg = layer(trg, src_enc, trg_mask, subsq_mask, src_trg_mask)
        
        return x


