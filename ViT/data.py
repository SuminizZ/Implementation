import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import os

from ViT.confgs import *

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    images_padded = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
    return images_padded, targets

    
def get_coco_dataloader(n_batch, coco_root, annFile):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
    ])

    coco_dataset = CocoDetection(root=coco_root, annFile=annFile, transform=transform)

    # valid_indices = []
    # for i in range(len(coco_dataset)):
    #     img_path = coco_dataset.coco.imgs[coco_dataset.ids[i]]['file_name']
    #     if os.path.exists(os.path.join(coco_dataset.root, img_path)):
    #         valid_indices.append(i)

    # valid_coco_dataset = torch.utils.data.Subset(coco_dataset, valid_indices)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    num_samples = len(coco_dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(coco_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=n_batch, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# print("hi")