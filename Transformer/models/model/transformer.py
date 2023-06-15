import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from model.encoder import *
from model.decoder import *


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, trg_vocab_size, hidden_dim, num_heads, N, device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(embed_dim, src_vocab_size, hidden_dim, num_heads, N, device)
        self.decoder = Decoder(embed_dim, trg_vocab_size, hidden_dim, num_heads, N, device)
        self.linear = nn.Linear(embed_dim, trg_vocab_size)

    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        src_pad_mask = self.make_pad_mask(src, src, src_pad_idx, src_pad_idx)
        trg_pad_mask = self.make_pad_mask(trg, trg, trg_pad_idx, trg_pad_idx)
        src_trg_pad_mask = self.make_pad_mask(src, trg, src_pad_idx, trg_pad_idx)
        trg_subsq_mask = self.make_subsq_mask(trg_pad_mask)

        src_enc = self.encoder(src, src_pad_mask)
        trg_out = self.decoder(trg, src_enc, trg_pad_mask, trg_subsq_mask, src_trg_pad_mask)
        trg_out = self.linear(trg_out)
        
        return trg_out

    def make_pad_mask(self, src, trg, src_pad_idx, trg_pad_idx):
        src_len = src.size(1)       # n_batch, src_len
        trg_len = trg.size(1)     # n_batch, trg_len 
        
        src = src.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)       
        src = src.repeat(1, 1, trg_len, 1)         # n_batch, 1, trg_len, src_len
        
        trg = trg.ne(self.trg_pad_idx).unsqueeze(1).unsqueeze(3)    
        trg = trg.repeat(1, 1, 1, src_len)        # n_batch, 1, trg_len, src_len

        pad_mask = src & trg       # n_batch, 1, trg_len, src_len

        return pad_mask

    def make_subsq_mask(self, trg_pad_mask):
        subsq_mask = torch.tril(trg_pad_mask).type(torch.BoolTensor)    
        
        return subsq_mask


