import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


class TokenEmbedding(nn.Embedding):
    """
    Operate Token Embedding
    """
    def __init__(self, embed_dim, vocab_size):
        """
        Inputs:
        - embed_dim : embedding dimension
        - vocab
        """
        super(TokenEmbedding, self).__init__(len(vocab_size), embed_dim, padding_idx=1)


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, device, max_len=500):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super(PositionalEncoding, self).__init__()
        
        # self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        pe = torch.zeros(1, max_len, embed_dim)
        pe.requries_grad = False

        p_seq = torch.arange(0, max_len).unsqueeze(1)
        p_idx = 10000**(torch.arange(0, embed_dim//2)*(-2/embed_dim))
        outer = p_seq*p_idx

        even_idx = torch.arange(0, embed_dim//2)*2
        odd_idx = torch.arange(0, embed_dim//2)*2 + 1

        pe[:, :, even_idx] = torch.sin(outer)
        pe[:, :, odd_idx] = torch.cos(outer)

        self.register_buffer('pe', pe)

    def forward(self, x):
        N, S, D = x.shape
        pe_output = x + self.pe[0, :S, :]
        return pe_output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """       
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.n_head = num_heads
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None, subsq_mask=None):
        N, S, E = query.shape
        N, T, E = value.shape      # S = T in self-attention
        output = torch.empty((N, S, E))
        D = E//self.n_head
        
        query = self.query(query).reshape((N, S, self.n_head, D)).permute(0, 2, 1, 3)
        key = self.key(key).reshape((N, T, self.n_head, D)).permute(0, 2, 3, 1)
        value = self.value(value).reshape((N, T, self.n_head, D)).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)/math.sqrt(D)   # N, H, S, T

        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (S-T,T-S), 'constant', 0)
            attention = attention.masked_fill(attn_mask == 0, -1e9)    # softmax 

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


class PointWiseFeedForward(nn.Modue):
    """
    Fully connected layer applied to each position identically after multi-head attention.
    """
    def __init__(self, embed_dim, hidden_dim, drop_prob=0.1):
        super(PointWiseFeedForward)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(drop_prob),
                                nn.Linear(hidden_dim, embed_dim))

    def forward(self, x):
        output = self.ff(x)
        return

