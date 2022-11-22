import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lossfx(loss, params):

    if loss == 'BCEWithLogitsLoss':
        lossfx = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params['pos_weight']))
    elif loss == 'DiceLoss':
        lossfx = DiceLoss()
    elif loss == 'focalloss':
        lossfx = FocalLoss(alpha=params['lossfx']['falpha'], gamma=params['lossfx']['fgamma'],
                           weight=params['lossfx']['fweight'])
    elif loss == 'iou':
        lossfx = IoULoss()
    elif loss == 'ioudice':
        lossfx = IoUDiceLoss(params, alpha=params['lossfx']['alpha'], beta=params['lossfx']['beta'])
    elif loss == 'ioufocal':
        lossfx = IoUFocalLoss(params, alpha=params['lossfx']['alpha'], beta=params['lossfx']['beta'])
    elif loss == 'ioudicefocal':
        lossfx = IoUDiceFocalLoss(params, alpha=params['lossfx']['alpha'], beta=params['lossfx']['beta'],
                                  gamma=params['lossfx']['gamma'])
    else:
        raise ValueError(f'No support loss function {loss}!')

    return lossfx


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        loss = 1 - dice

        return torch.log((torch.exp(loss) + torch.exp(-loss)) / 2.0)


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)

        if self.weight is not None:
            self.weight = torch.tensor(self.weight).to(inputs.device)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.weight, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-bce_loss)
        F_loss = at*(1-pt)**self.gamma * bce_loss

        return F_loss.mean()


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class IoUDiceLoss(nn.Module):
    def __init__(self, params, alpha=0.5, beta=0.5):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.iou = IoULoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        iou_loss = self.iou(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return self.alpha*iou_loss + self.beta*dice_loss


class IoUFocalLoss(nn.Module):
    def __init__(self, params, alpha=0.5, beta=0.5):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.iou = IoULoss()
        self.focal = FocalLoss(alpha=params['lossfx']['falpha'], gamma=params['lossfx']['fgamma'])

    def forward(self, inputs, targets):
        iou_loss = self.iou(inputs, targets)
        focal_loss = self.focal(inputs, targets)

        return self.alpha*iou_loss + self.beta*focal_loss


class IoUDiceFocalLoss(nn.Module):
    def __init__(self, params, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iou = IoULoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=params['lossfx']['falpha'], gamma=params['lossfx']['fgamma'])

    def forward(self, inputs, targets):
        iou_loss = self.iou(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)

        return self.alpha*iou_loss + self.beta*dice_loss + self.gamma*focal_loss

