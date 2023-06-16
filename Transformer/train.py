import torch
import torch.nn as nn
import time 
from tqdm import tqdm

from data import *
from config import *
from model.transformer import Transformer


criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
model = Transformer(embed_dim, 
                    src_vocab_size, trg_vocab_size, 
                    src_pad_idx, trg_pad_idx, 
                    hidden_dim, num_heads, N, device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=eps)


def train(model, iterator, criterion, clip, epochs=100, print_every=100):

    model.train()
    train_loss_history = []

    for e in range(epochs):
        e_loss = run_epoch(iterator, model, criterion, clip, optimizer)
        train_loss_history += [e_loss]

    return train_loss_history

def run_epoch(iterator, model, criterion, clip, optimizer=None):
    
    e_loss = 0
    
    for batch in tqdm(iterator):
        src = batch.src
        trg = batch.trg
        
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        b_loss = criterion(output, trg)

        optimizer.zero_grad()
        b_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        e_loss += b_loss.item()

    return e_loss


