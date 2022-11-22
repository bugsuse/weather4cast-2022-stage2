import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(y_hat, params):

    if params.get('scale_factor') is not None:
        y_hat = y_hat.squeeze()
        bs, ts, width, height = y_hat.shape
        height = int(height * params['scale_factor'])
        width = int(width * params['scale_factor'])
        tar_size = (height, width)
        y_hat = F.interpolate(y_hat, size=tar_size, mode='bilinear')
        y_hat = y_hat.unsqueeze(dim=1)

    return y_hat

