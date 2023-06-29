import torch

n_batch = 8
num_workers = 2
pin_memory = True

num_grids = 7
num_bboxes = 2
num_classes = 20
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

wd = 0
epochs = 1000
lr = 2e-5

iou_threshold = 0.5
threshold = 0.3