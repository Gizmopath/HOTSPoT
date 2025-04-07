import torch
import contextlib
from tqdm import tqdm

def run_epoch(model, loader, criterion, num_classes, optimizer=None, scaler=None, is_train=True):
    model.train() if is_train else model.eval()
    epoch_loss, overall_ious, overall_accuracies = 0.0, [], []
    class_ious = [[] for _ in range(num_classes)]
    class_accuracies = [[] for _ in range(num_classes)]

    loop = tqdm(loader, leave=False)
    loop.set_description("Training" if is_train else "Validation")

    with torch.no_grad() if not is_train else contextlib.nullcontext():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(images)
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)

            if is_train:
                scaler.scale(loss).backward() if scaler else loss.backward()
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            overall_iou, overall_accuracy, ious, accuracies = calculate_metrics_per_class(outputs, masks, num_classes)
            overall_ious.append(overall_iou)
            overall_accuracies.append(overall_accuracy)
            for cls in range(num_classes):
                class_ious[cls].append(ious[cls])
                class_accuracies[cls].append(accuracies[cls])

            loop.set_postfix(loss=loss.item())

    mean_overall_iou = np.mean(overall_ious).tolist()
    mean_overall_accuracy = np.mean(overall_accuracies).tolist()
    mean_class_ious = [np.mean(cls_ious).tolist() for cls_ious in class_ious]
    mean_class_accuracies = [np.mean(cls_accs).tolist() for cls_accs in class_accuracies]

    return epoch_loss / len(loader), mean_overall_iou, mean_overall_accuracy, mean_class_ious, mean_class_accuracies
