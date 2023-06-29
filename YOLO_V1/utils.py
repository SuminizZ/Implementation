import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def plot_image(image, boxes):
    im = np.array(image)
    h, w, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        assert len(box) == 4, "Got more values than x, y, w, h"
        x1 = box[0] - box[2] / 2   
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle(
                                (x1*w, y1*h),
                                box[2]*w,
                                box[3]*h,
                                linewidth=1,
                                edgecolor="r",
                                facecolor="none",
                            )

        ax.add_patch(rect)
    plt.show()


def run_one_batch(n_batch, idx, loader, model, iou_threshold, threshold, device):
    pred_boxes, gt_boxes = [], []
    model = model.to(device)
    model.eval()       # make sure to turn off the train mode before getting final bboxes. 

    img, labels = next(iter(loader))
    train_idxs = torch.arange(n_batch).unsqueeze(-1).unsqueeze(-1).repeat(1, 49, 1)

    imgs = img.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = model(img)   

    S = 7
    n_batch = len(preds)
    pred_bboxes = convert_cellboxes(preds).reshape(n_batch, S*S, -1)
    gt_bboxes = convert_cellboxes(labels).reshape(n_batch, S*S, -1)

    pred_bboxes = torch.concat([train_idxs, pred_bboxes], dim=-1)
    gt_bboxes = torch.concat([train_idxs, gt_bboxes], dim=-1)

    pred_boxes += non_max_suppression(pred_bboxes, iou_threshold, threshold, "center")
    for bb in [box[np.where(box[:, 2] > threshold)[0]] for box in gt_bboxes]:
        gt_boxes += [b for b in bb]
    
    plot_boxes = [box[3:] for box in pred_boxes if box[0] == idx]
    print(plot_boxes)
    plot_image(img[idx].permute(1,2,0).to("cpu"), plot_boxes)
            

def compute_ious(box_format, pred_bboxes, gt_bboxes):
    '''
    calculates intersection over union between predicted bboxes and ground truth bboxes 
    box_format 
        - center : (center_x, center_y, w, h)
        - corner : (x1, y1, x2, y2)
    '''
    if box_format == 'center':
        px1 = pred_bboxes[..., 0:1] - (pred_bboxes[..., 2:3] / 2)
        py1 = pred_bboxes[..., 1:2] - (pred_bboxes[..., 3:4] / 2)
        px2 = pred_bboxes[..., 2:3] - (pred_bboxes[..., 2:3] / 2)
        py2 = pred_bboxes[..., 3:4] - (pred_bboxes[..., 3:4] / 2)
        gtx1 = gt_bboxes[..., 0:1] - (gt_bboxes[..., 2:3] / 2)  
        gty1 = gt_bboxes[..., 1:2] - (gt_bboxes[..., 3:4] / 2)
        gtx2 = gt_bboxes[..., 2:3] - (gt_bboxes[..., 2:3] / 2)
        gty2 = gt_bboxes[..., 3:4] - (gt_bboxes[..., 3:4] / 2)

    elif box_format == 'corner':
        px1 = pred_bboxes[..., 0:1]
        py1 = pred_bboxes[..., 1:2]
        px2 = pred_bboxes[..., 2:3]
        py2 = pred_bboxes[..., 3:4]
        gtx1 = gt_bboxes[..., 0:1]
        gty1 = gt_bboxes[..., 1:2]
        gtx2 = gt_bboxes[..., 2:3]
        gty2 = gt_bboxes[..., 3:4]

    x1 = torch.max(px1, gtx1)
    y1 = torch.max(py1, gty1)
    x2 = torch.min(px2, gtx2)
    y2 = torch.max(py2, gty2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)   # in case two bboxes are not overlapped. 
    total = abs((px2 - px1)*(py2 - py1) + (gtx2 - gtx1)*(gty2 - gty1) - intersection)
    ious = intersection / (total + 1e-8)
    return ious


def non_max_suppression(bboxes, iou_threshold, threshold, box_format='center'):
    '''
    bboxes : (n_batch, 7*7, [pred_clss, best_confidence_score, coordinates (x, y, w, h)])
    iou_threshold != threshold
    threshold here refers to the minimum confidence score to be selected as true bbox.
    '''
    n_batch = len(bboxes)
    post_nms_bboxes = []
    for j in range(n_batch):
        j_bboxes = bboxes[j]
        j_bboxes = j_bboxes[np.where(j_bboxes[:, 2] > threshold)[0]]
        
        if not j_bboxes.size(0):
            continue
        
        confidence_scores = j_bboxes[:, 2]
        order_idxs = confidence_scores.ravel().argsort(descending=True)    # sort bboxes by confidence scores in a descending order.

        keep_bboxes = []
        order_idxs = order_idxs.argsort(descending=True)
        while (order_idxs.size(0) > 0):
            target_idx = order_idxs[0]
            keep_bboxes.append(target_idx.item())

            ious = compute_ious(box_format, j_bboxes[order_idxs[1:]], j_bboxes[target_idx])
            keep_idxs = np.where(ious <= iou_threshold)[0]
            order_idxs = order_idxs[keep_idxs+1]

        post_nms_bboxes += [box for box in j_bboxes[keep_bboxes]]

    return post_nms_bboxes


def convert_cellboxes(predictions, S=7):
    """
    Convert output of yolo v1
    (n_batch, 7*7*(5*B+C)) -> (n_batch, 7, 7, [pred_clss, best_confidence_score, center coordinates])
    - Convert bounding boxes with grid split size S relative to cell ratio into entire image ratio.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]    # 1 bbox 
    bboxes2 = predictions[..., 26:30]    # 2 bbox 
    
    # select best bounding box with highest confidence score among 2 candidates
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)   # 2 x (n_batch x 7 x 7)
    best_box = scores.argmax(0).unsqueeze(-1)     # n_batch x 7 x 7 x 1
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2  

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = (1/S) * (best_boxes[..., :1] + cell_indices)    # 0 + x*(1/S), 1 + x*(1/S), 2 + x*(1/S), ...  6 + x*(1/S)
    y = (1/S) * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  
    w_y = (1/S) * best_boxes[..., 2:4]  
     
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    pred_class = predictions[..., :20].argmax(-1).unsqueeze(-1)      # class with best pred scores. 
    best_conf = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)    # best confidence score 
    converted_preds = torch.cat((pred_class, best_conf, converted_bboxes), dim=-1)
    return converted_preds


def get_bboxes(n_batch, loader, model, iou_threshold, threshold, device, box_format='center'):
    pred_boxes, gt_boxes = [], []
    model = model.to(device)
    model.eval()   # make sure to turn off the train mode before getting final bboxes. 

    for batch_idx, (img, labels) in enumerate(loader):
        train_idxs = (torch.arange(n_batch) + (batch_idx*n_batch)).unsqueeze(-1).unsqueeze(-1).repeat(1, 49, 1)

        img = img.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(img)   # (n_batch, S*S*(C+B*5))

        S = 7
        n_batch = len(preds)
        pred_bboxes = convert_cellboxes(preds).reshape(n_batch, S*S, -1)
        gt_bboxes = convert_cellboxes(labels).reshape(n_batch, S*S, -1)

        pred_bboxes = torch.concat([train_idxs, pred_bboxes], dim=-1)
        gt_bboxes = torch.concat([train_idxs, gt_bboxes], dim=-1)

        pred_boxes += non_max_suppression(pred_bboxes, iou_threshold, threshold, "center")
        for bb in [box[np.where(box[:, 2] > threshold)[0]] for box in gt_bboxes]:
            gt_boxes += [b for b in bb]
        # if batch_idx == 10: break

    return pred_boxes, gt_boxes


def get_average_precision(TP, FP, num_T):
    eps = 1e-8
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (num_T + eps)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    # torch.trapz for numerical integration
    ap = torch.trapz(precisions, recalls)

    return ap.item()


def mean_average_precision(pred_boxes, gt_boxes, iou_threshold, num_classes, box_format="center"):
    '''
    Args:
    pred_boxes = tensor([train_idx, pred_class, best_confidence_score, x, y, w, h])
    gt_boxes = same as pred_boxes
    '''
    average_precisions = []
    for c in range(num_classes):
        c_pred_boxes = [box.unsqueeze(0) for box in pred_boxes if box[1] == c]    # check if class matched, shape : (num_detections, 7)
        c_gt_boxes = [box.unsqueeze(0) for box in gt_boxes if box[1] == c]

        num_T, num_P = len(c_gt_boxes), len(c_pred_boxes)
        if not num_T: continue
        if not num_P:
            average_precisions.append(0)
            continue

        c_pred_boxes = torch.concat(c_pred_boxes, dim=0)
        FP = torch.zeros(num_P)
        TP = torch.zeros(num_P)

        gt_cnt = Counter([int(gt[:, 0].item()) for gt in c_gt_boxes])   # counting gt boxes by train_idx (per image) within a class, gt_cnt = {0:3, 1:5}
        gt_cnt_graph = {}
        # set 1 if detected -> only one predicter is responsible for each object (gt box)
        for idx, cnt in gt_cnt.items():
            gt_cnt_graph[idx] = torch.zeros(cnt)

        c_gt_boxes = torch.concat(c_gt_boxes, dim=0)
        for pred_idx, box in enumerate(c_pred_boxes):
            train_idx = box[0].item()      # check if train_idx matched
            target_gt_idxs = np.where(c_gt_boxes[:, 0] == train_idx)[0]

            if not len(target_gt_idxs):
                FP[pred_idx] = 1
                continue

            ious = compute_ious('center', c_gt_boxes[target_gt_idxs, 3:], box[3:])
            best_iou_idx = ious.argmax(dim=0)
            best_iou = ious[best_iou_idx]

            # check 1. whether current gt_box is already detected by other pred box and 2. object detection clearly captures ground truth.
            if gt_cnt_graph[int(train_idx)][best_iou_idx] == 0 and best_iou >= iou_threshold:
                TP[pred_idx] = 1
                gt_cnt_graph[int(train_idx)][best_iou_idx] = 1
            else:
                FP[pred_idx] = 1

        average_precisions.append(get_average_precision(FP, TP, num_T))

    return sum(average_precisions)/len(average_precisions)
            

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



