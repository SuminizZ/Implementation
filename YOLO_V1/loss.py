"""
Implementation of Yolo Loss Function from the original yolo paper

"""
import torch
import torch.nn as nn
from utils import compute_ious

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        ## coordinate regression loss 

        # calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = compute_ious('center', predictions[..., 21:25], target[..., 21:25])
        iou_b2 = compute_ious('center', predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction -> one bounding box prediction per each object 
        iou_maxes, bestbox = torch.max(ious, dim=0)     # val, idx (argmax) = 0 (1st bbox) or 1 (2nd bbox) 
        obj_mask = target[..., 20:21]                   # object / no object (whether ground truth box holds object or not)

        # Set boxes with no object in them to 0. Only select the box with max iou with ground truth 
        box_predictions = obj_mask*(bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        box_targets = obj_mask*target[..., 21:25]

        # Take sqrt of width, height of boxes
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))
        
        ## object loss 
        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]   
        object_loss = self.mse(
            torch.flatten(obj_mask * pred_box),
            torch.flatten(obj_mask * target[..., 20:21]),
        )
        
        ## no object loss 
        no_object_loss = self.mse(torch.flatten((1 - obj_mask) * predictions[..., 20:21], start_dim=1),
                                  torch.flatten((1 - obj_mask) * target[..., 20:21], start_dim=1))

        no_object_loss += self.mse(torch.flatten((1 - obj_mask) * predictions[..., 25:26], start_dim=1),
                                   torch.flatten((1 - obj_mask) * target[..., 20:21], start_dim=1))
        
        ## classification loss 
        class_loss = self.mse(torch.flatten(obj_mask * predictions[..., :20], end_dim=-2,),
                              torch.flatten(obj_mask * target[..., :20], end_dim=-2,))

        loss = ( self.lambda_coord * box_loss  
               + object_loss  
               + self.lambda_noobj * no_object_loss 
               + class_loss)

        return loss

