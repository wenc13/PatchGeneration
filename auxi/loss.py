import torch
from extension.chamfer_distance.chamfer_distance import ChamferDistance


def ChamferLoss(target, prediction, reduction='mean'):
    dist1, dist2, _, _ = ChamferDistance()(target, prediction)

    if reduction == 'mean':
        loss = torch.mean(dist1) + torch.mean(dist2)
    elif reduction == 'sum':
        loss = torch.sum(dist1) + torch.sum(dist2)
    else:
        raise ValueError()

    return loss
