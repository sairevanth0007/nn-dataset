import torch

def compute(predictions, targets):
    """
    Compute mean Average Precision for object detection
    Args:
        predictions: List of dictionaries with 'boxes', 'labels', 'scores'
            - boxes: tensor (N, 4) in [x1,y1,x2,y2] format
            - labels: tensor (N,) of class ids
            - scores: tensor (N,) of confidence scores
        targets: List of dictionaries with 'boxes', 'labels'
            - boxes: tensor (M, 4) in [x1,y1,x2,y2] format
            - labels: tensor (M,) of class ids
    Returns:
        tuple: (map_score, 1)
    """
    iou_threshold = 0.5
    all_aps = []
    for class_id in range(91):
        class_preds = []
        class_targets = []
        for pred, target in zip(predictions, targets):
            mask = pred['labels'] == class_id
            boxes = pred['boxes'][mask]
            scores = pred['scores'][mask]
            gt_mask = target['labels'] == class_id
            gt_boxes = target['boxes'][gt_mask]
            class_preds.append((boxes, scores))
            class_targets.append(gt_boxes)
        if not any((len(gt) > 0 for gt in class_targets)):
            continue
        ap = compute_ap(class_preds, class_targets, iou_threshold)
        all_aps.append(ap)
    if len(all_aps) == 0:
        return (0.0, 1)


    return (float(torch.tensor(all_aps).mean()), 1)

def compute_ap(class_preds, class_targets, iou_threshold):
    """
    Compute Average Precision for a single class
    """
    boxes_all = []
    scores_all = []
    matches = []
    n_gt = 0
    for (boxes, scores), gt_boxes in zip(class_preds, class_targets):
        if len(gt_boxes) > 0:
            n_gt += len(gt_boxes)
        if len(boxes) == 0:
            continue
        iou = box_iou(boxes, gt_boxes)
        for i in range(len(boxes)):
            if len(gt_boxes) == 0:
                matches.append(0)
            else:
                matches.append(1 if iou[i].max() >= iou_threshold else 0)
            scores_all.append(scores[i])
    if not scores_all:
        return 0.0
    scores_all = torch.tensor(scores_all)
    matches = torch.tensor(matches)
    sorted_indices = torch.argsort(scores_all, descending=True)
    matches = matches[sorted_indices]
    tp = torch.cumsum(matches, dim=0)
    fp = torch.cumsum(~matches.bool(), dim=0)
    precision = tp / (tp + fp)
    recall = tp / n_gt if n_gt > 0 else torch.zeros_like(tp)
    ap = 0.0
    for r in torch.linspace(0, 1, 11):
        mask = recall >= r
        if mask.any():
            ap += precision[mask].max() / 11
    return ap

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union
