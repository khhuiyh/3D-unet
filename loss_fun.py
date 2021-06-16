import torch.nn as nn
import torch


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        batch_num = targets.size(0)
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - (IoU / batch_num)


class IoULoss_plus_BCEloss(nn.Module):
    def __init__(self, smooth=1.0):  # 200
        super(IoULoss_plus_BCEloss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        batch_num = targets.size(0)
        m1 = inputs.reshape(batch_num, -1)
        m2 = targets.reshape(batch_num, -1)

        intersection = (m1 * m2).sum(1)
        union = m1.sum(1) + m2.sum(1) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        score = -torch.log_(score)
        IoUloss = score.sum() / batch_num
        bce = nn.BCELoss()
        bceloss = bce(inputs, targets)

        return IoUloss + bceloss


class IoULoss_plus_mseloss(nn.Module):
    def __init__(self, smooth=1.0, k=0.5):
        super(IoULoss_plus_mseloss, self).__init__()
        self.smooth = smooth
        self.k = k

    def forward(self, inputs, targets):
        batch_num = targets.size(0)
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        mse = nn.MSELoss()
        mse_loss = mse(inputs, targets)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)
        return self.k * (1 - (IoU / batch_num)) + (1 - self.k) * mse_loss


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1.0):
        super(IoU, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        batch_num = targets.size(0)
        m1 = inputs.reshape(batch_num, -1)
        m2 = targets.reshape(batch_num, -1)

        intersection = m1 * m2
        score = (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + self.smooth)

        return score


class diceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1.0):
        super(diceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        batch_num = targets.size(0)
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        return 1 - (dice / batch_num)

# class IoULoss_plus_BCE(nn.Module):
#     def __init__(self, smooth=1.0, k=0.5, k1iou=0.0):
#         super(IoULoss_plus_BCE, self).__init__()
#         self.smooth = smooth
#         self.k = k
#         self.k1iou = k1iou
#
#     def forward(self, inputs, targets):
#         batch_num = targets.size(0)
#
#         # flatten label and prediction tensors
#         inputs = inputs.contiguous().view(-1)
#         targets = targets.contiguous().view(-1)
#
#         # intersection is equivalent to True Positive count
#         # union is the mutually inclusive area of all labels & predictions
#         kl = targets.sum() / targets.size(0)
#         if kl.cpu().item() > self.k1iou:
#             k_positiveiou = kl
#         else:
#             k_positiveiou = self.k1iou
#         bce = nn.BCELoss()
#         bce_loss = bce(inputs, targets)
#
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection
#
#         IoUloss = (intersection + self.smooth) / (union + self.smooth)
#
#         inputs = 1 - inputs
#         targets = 1 - targets
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection
#
#         IoUloss = (intersection + self.smooth) / (union + self.smooth) * (1 - k_positiveiou) + IoUloss * k_positiveiou
#         return self.k * (1 - (IoUloss / batch_num)) + (1 - self.k) * bce_loss

