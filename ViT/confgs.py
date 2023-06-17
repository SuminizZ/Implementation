import torch
import torch.nn as nn

n_batch = 64
embed_dim = 1024
num_heads = 16
num_layers = 8

p = 32
input_dim = 3*(p**2)
hidden_dim = 1024

wd = 0.1
lr = 0.001
dropout_prob = 0.0

warmup_steps = 3
epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)