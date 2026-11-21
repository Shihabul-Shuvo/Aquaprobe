import torch
import torch.nn.functional as F


def dice_loss(logits, target, num_classes, ignore_index=255):
    valid = (target != ignore_index).unsqueeze(1).to(logits.dtype)
    prob = logits.softmax(1) * valid
    onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).to(logits.dtype) * valid
    intersection = (prob * onehot).sum((0, 2, 3))
    denominator = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


class SegmentationLoss(torch.nn.Module):
    def __init__(self, num_classes, lambda_dice=1.0, lambda_route=1.0, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_route = lambda_route
        self.ignore_index = ignore_index

    def forward(self, logits, route_logits, target, cluster):
        logits = logits.float()
        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        loss = loss + self.lambda_dice * dice_loss(logits, target, self.num_classes, self.ignore_index)
        if route_logits is not None and (cluster >= 0).any():
            routed = cluster >= 0
            loss = loss + self.lambda_route * F.cross_entropy(route_logits[routed].float(), cluster[routed])
        return loss
