import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from ViT.vit_layers import *
from ViT.embeddings import *

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, device, drop_prob=0.1):
        """
        Basic Encoder block constists of 1 MSA and 1 MLP
        """
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention (MSA)
        self.layernorm1 = LayerNorm(embed_dim)
        self.msa = MSA(embed_dim, num_heads)

        # Point-Wise Feed Forward 
        self.layernorm2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, data):
        msa_out = self.layernorm1(data)
        msa_out = self.msa(msa_out, msa_out, msa_out)
        msa_out += data
        
        mlp_out = self.layernorm2(msa_out)
        mlp_out = self.mlp(msa_out)
        mlp_out = self.dropout(mlp_out + msa_out)

        return mlp_out


class ViT(nn.Module):

    def __init__(self, n_batch, input_dim, embed_dim,
                 num_classes, pool,
                 hidden_dim, num_heads, 
                 num_layers, device):
        super(Encoder, self).__init__()

        # Embedding & Encoding
        self.patchify = PatchEmbedding(p, input_dim, embed_dim)
        self.class_token = ClassTokenEmbedding(n_batch, embed_dim)
        self.pos_enc = PositionalEmbedding(N, embed_dim)

        # Stacks of encoder blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])

        # final classification head 
        self.classhead = ClassificationHead(embed_dim, num_classes, pool)

    def forward(self, data):
        data = self.patchify(data)
        data = self.class_token(data)
        data = self.pos_enc(data)

        for layer in self.encoder_blocks:
            data = layer(data)

        classhead = self.classhead(data)
        
        return data



