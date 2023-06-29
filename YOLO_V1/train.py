import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader

import time
from tqdm import tqdm

from model import YOLOv1
from data import VOCDataset
from utils import *
# from confgs import *
from loss import YoloLoss


n_batch = 8
num_workers = 2
pin_memory = True
num_grids = 7
num_bboxes = 2
num_classes = 20
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
wd = 0
epochs = 100
lr = 2e-5
iou_threshold = 0.5
threshold = 0.35
# load_model = True


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def plot_image(idx, model):
    csv_file = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/100examples.csv'
    img_dir = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/images'
    label_dir = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/labels'

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

    train_dataset = VOCDataset(csv_file, img_dir, label_dir, transform=transform)

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=n_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=True,
        )

    run_one_batch(n_batch, idx, train_loader, model, iou_threshold, threshold, device)
    return


def compute_loss(loader, model, optimizer, yolo_loss):
    model.train()
    model = model.to(device)

    loop = tqdm(loader, leave=True)
    loss_history = []
    for batch_idx, (img, labels) in enumerate(loop):
        img, labels = img.to(device), labels.to(device)
        preds = model(img)
        loss = yolo_loss(preds, labels)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    mean_loss = sum(loss_history)/len(loss_history)
    print(f"  Mean loss : {mean_loss}")
    return mean_loss

def main(load_model):
    csv_file = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/100examples.csv'
    img_dir = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/images'
    label_dir = '/content/drive/MyDrive/Implement/YOLO_V1/dataset/labels'
    chkpt_dir = '/content/drive/MyDrive/Implement/YOLO_V1/checkpoints/my_checkpoint.pth.tar'
    load_dir = chkpt_dir

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

    train_dataset = VOCDataset(csv_file, img_dir, label_dir, transform=transform)

    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    yolo_loss = YoloLoss()

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=n_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=True,
        )

    if load_model:
        load_checkpoint(torch.load(load_dir), model, optimizer)
        return model, optimizer

    prev_loss = 9999
    early_stopping = 0
    for e in range(epochs):

        pred_boxes, gt_boxes = get_bboxes(n_batch, train_loader, model, iou_threshold, threshold, device)

        mAP = mean_average_precision(pred_boxes, gt_boxes, iou_threshold, num_classes)
        
        print(f"Train mAP: {mAP}")

        mean_loss = compute_loss(train_loader, model, optimizer, yolo_loss)

        if mean_loss <= prev_loss:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=chkpt_dir)
            time.sleep(7)
        else:
            early_stopping += 1
        
        if early_stopping == 2 :
            print("----- Early Stopping -----")
            break

        prev_loss = mean_loss

    return



