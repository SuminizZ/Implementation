import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    
    def __init__(self, p, input_dim, embed_dim):
        super(PatchEmbedding, self).__init__()
        """
        Embedding image input (n_batch, channel, H, W) into (n_batch, N, (P*P*C)) where N = H*W/P*P*C
        Args :
            - p : patch size
        """
        self.p = p
        self.patch_embedding = nn.Sequential(Rearrange('b c (h1 p) (w1 p) -> b (h1 w1) (c p p)', p = self.p),
                                             nn.Linear(input_dim, embed_dim))

    def forward(self, x):
        x = self.patch_embedding(x)
        return x 


class ClassTokenEmbedding(nn.Module):

    def __init__(self, n_batch, embed_dim):
        super(PatchEmbedding, self).__init__()
        """
        Add classfication token to the sequence of embedded patches. (n_batch, N, embed_dim) -> (n_batch, N+1, embed_dim)
        Args :
            - n_batch : batch size
            - embed_dim : patch embedded dimension 
        """
        self.classtoken = nn.Parameter(torch.randn(n_batch, 1, embed_dim))

    def forward(self, x):
        
        return torch.cat([x, self.classtoken], dim=1)


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_dim, max_len=5000):
        """
        Construct the PositionalEncoding layer.
        Args:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super(PositionalEncoding, self).__init__()
        
        self.pos_emb = nn.Parameter(torch.randn(max_len, embed_dim))      

    def forward(self, x):
        n_batch, N, embed_dim = x.shape
        pe_output = x + self.pos_emb[:N, :]
        return pe_output



