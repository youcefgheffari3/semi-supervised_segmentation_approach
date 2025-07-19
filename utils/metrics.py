import numpy as np


def compute_iou(pred, target, num_classes=21, ignore_index=255):
    """
    Compute Intersection over Union (IoU) for a single image.

    Args:
        pred (torch.Tensor or np.ndarray): Model predicted mask [H, W].
        target (torch.Tensor or np.ndarray): Ground-truth mask [H, W].
        num_classes (int): Number of segmentation classes.
        ignore_index (int): Ignore this label in ground-truth (usually 255 for VOC).

    Returns:
        List[float]: IoU score for each class. NaN if class not present.
    """
    pred = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
    target = target.cpu().numpy() if hasattr(target, 'cpu') else target

    pred = pred.flatten()
    target = target.flatten()

    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        # Exclude ignore_index pixels from union/intersection
        valid_mask = target != ignore_index
        pred_inds &= valid_mask
        target_inds &= valid_mask

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            ious.append(float('nan'))  # Class absent in GT and predictions
        else:
            ious.append(intersection / union)

    return ious


def compute_mean_iou(all_ious):
    """
    Compute mean IoU across dataset (ignores NaN values per class).

    Args:
        all_ious (List[List[float]]): List of IoUs per image.

    Returns:
        np.ndarray: Per-class mean IoU.
        float: Mean IoU over all valid classes.
    """
    ious_array = np.array(all_ious)
    mean_per_class = np.nanmean(ious_array, axis=0)  # Mean IoU for each class
    mIoU = np.nanmean(mean_per_class)                # Overall mean IoU

    return mean_per_class, mIoU
