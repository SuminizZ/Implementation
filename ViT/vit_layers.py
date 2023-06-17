import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Construct a new MultiHeadAttention layer.
        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """       
        super(MSA, self).__init__()
        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.n_head = num_heads
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, pad_mask=None, subsq_mask=None):
        N, S, E = query.shape
        N, T, E = value.shape      # S = T in self-attention
        output = torch.empty((N, S, E))
        D = E//self.n_head
        
        query = self.query(query).reshape((N, S, self.n_head, D)).permute(0, 2, 1, 3)
        key = self.key(key).reshape((N, T, self.n_head, D)).permute(0, 2, 3, 1)
        value = self.value(value).reshape((N, T, self.n_head, D)).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)/math.sqrt(D)   # N, H, S, T

        if pad_mask is not None:
            attention = attention.masked_fill(pad_mask == 0, -1e9)    # softmax 

        if subsq_mask is not None:
            attention = attention.masked_fill(subsq_mask == 0, -1e9)

        attention_prob = F.softmax(attention, dim=-1)

        output = torch.matmul(attention_prob, value)
        output = output.transpose(1,2).contiguous().view(N, S, E)
        
        return output


class LayerNorm(nn.Module):
    """
    Normalize across all features. 
    """
    def __init__(self, embed_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-9
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        x = (x - x.mean(-1, keepdims=True))/(x.std(-1, keepdim=True) + self.eps)
        out = self.gamma*x + self.beta
        return out


class MLP(nn.Module):
    """
    MLP applied to each position identically after multi-head attention.
    """
    def __init__(self, embed_dim, hidden_dim, drop_prob=0.1):
        super(MLP, self).__init__()
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                nn.GELU(),
                                nn.Dropout(drop_prob),
                                nn.Linear(hidden_dim, embed_dim),
                                nn.Dropout(drop_prob))

    def forward(self, x):
        output = self.ff(x)
        return


class ClassificationHead(nn.Module):
    """
    Final MLP to get classification head : eithr mean or first element 
    """
    def __init__(self, embed_dim, num_classes, pool):
        super(ClassificationHead, self).__init__()

        self.pool = pool
        self.layernorm = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(self.layernorm,
                                 nn.Linear(embed_dim, num_classes))

    def forward(self,x):
        """
        Args
            - x : output of encoder (n_batch, N, embed_dim)
        """
        classhead = x.mean(dim=1) if self.pool == 'mean' else x[:, 0, :]

        classhead = self.mlp(classhead)
        return classhead
