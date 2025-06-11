import torch

def calculate_metrics_per_class(pred, target, num_classes):
    pred = pred.argmax(dim=1)
    ious, accuracies = [], []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        iou = intersection / union if union > 0 else 0
        accuracy = ((pred == cls) == (target == cls)).float().mean()
        ious.append(iou.item())
        accuracies.append(accuracy.item())
    overall_iou = sum(ious) / num_classes
    overall_accuracy = sum(accuracies) / num_classes
    return overall_iou, overall_accuracy, ious, accuracies
