import torch
import torch.nn as nn
from torch.nn import functional as F
import time 
import copy
from tqdm import tqdm

from data import *
from config import *
from model.transformer import Transformer


def translate(model, iterator, criterion, trg_sos_idx, trg_end_idx):
    
    loss = 0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            trg = batch.trg   # n_batch, trg_seq_len
            dec_trg = copy.deepcopy(trg)
            dec_trg[:, :] = 0 
            dec_trg[:, 0] = trg_sos_idx

            trg_seq_len = dec_trg.size(1)
            
            for i in range(1, trg_seq_len):
                dec_output = model(src, dec_trg[:, :i])   # n_batch, trg_seq_len, d_vocab

                pred_idx = F.softmax(dec_output, dim=-1)
                pred_idx = pred_idx.argmax(dim=-1)

                dec_trg[:, i] = pred_idx[:, -1]

                if pred_idx == trg_end_idx:
                    break

            dec_trg = dec_trg.contiguous().view(-1, dec_trg.shape[-1])
            trg = trg.contiguous().view(-1)

            b_loss = criterion(dec_trg[:, 1:], trg[:, 1:])
            loss += b_loss.item()

    return loss 





