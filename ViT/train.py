import torch
import torch.nn as nn
from torch.optim import *
from torch.nn import functional as F

import math
import numpy as np
import os 
from tqdm import tqdm
import wandb

from ViT.confgs import *
from ViT.data import *
from ViT.model import ViT
from utils.scheduler import *


def validate(val_loader, model, device):
    model.eval()

    print("Test Starts")

    criterion = nn.CrossEntropyLoss()
    classes = val_loader.dataset.label_names

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    n_correct, total = 0,0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for step, (x_batch, y_batch) in pbar[:3]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            _, preds = torch.max(outputs, 1)

            for label, pred in zip(y_batch, preds):
                if label == pred:
                    correct_pred[classes[label-1]] += 1
                    n_correct += 1
                total_pred[classes[label-1]] += 1
                total += 1

            description = f'Validation Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(n_correct/total, 4)}'
            pbar.set_description(description)
    
            # wandb logging
            wandb.log(
                {   
                    'Validation Loss': round(loss.item(), 4),
                    'Validation Accuracy': round(n_correct/total, 4)
                }
            )


def train(train_loader, val_loader, epochs, device, total_len):
    for epoch in range(1, epochs):
        model.train() 
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for step, (x_batch, y_batch) in pbar[:3]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            description = f'Epoch: {epoch}/{epochs} || Step: {step+1}/{len(train_loader)} || Training Loss: {round(loss.item(), 4)}'
            pbar.set_description(description)

            scheduler.step()
            wandb.log(
                {
                    "Learning Rate": optimizer.param_groups[0]['lr']
                }
            )

        validate(val_loader, model, device)


def main():
    data_dir = "/content/drive/MyDrive/Implement/datasets/coco_dataset"
    coco_root = os.path.join(data_dir, 'images/train2017')
    annFile = os.path.join(data_dir, 'annotations/annotations/instances_train2017.json')
    train_loader, val_loader, test_loader = get_coco_dataloader(n_batch, coco_root, annFile)

    criterion = nn.CrossEntropyLoss()

    model = ViT(n_batch, input_dim, embed_dim, num_classes, pool,
                hidden_dim, num_heads, num_layers, device)
    model.to(device)

    warmup_steps = 10000  
    total_steps = 7 * len(coco_dataloader)  
    scheduler = WarmupCosineDecay(optimizer, warmup_steps, total_steps)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd, eps=epochs)


if __name__ == "__main__":
    main()