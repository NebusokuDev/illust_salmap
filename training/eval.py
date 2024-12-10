import torch


def normalize01(tensor):
    return (tensor + 1) / 2

def eval_jaccard_index(saliency_map, ground_truth, threshold=0.05):
    saliency_map = normalize01(saliency_map)
    ground_truth = normalize01(ground_truth)
    intersection = torch.logical_and(saliency_map >= threshold, ground_truth >= threshold).sum()
    union = torch.logical_or(saliency_map >= threshold, ground_truth >= threshold).sum()
    return intersection / union

def eval_pixel_wise_accuracy(saliency_map, ground_truth, threshold=0.1):
    saliency_map = normalize01(saliency_map)
    ground_truth = normalize01(ground_truth)
    pixel_errors = torch.abs(saliency_map - ground_truth)

    correct_pixels = pixel_errors <= threshold

    accuracy = correct_pixels.float().mean()

    return accuracy