import torch
import torch.nn as nn


def mask_evaluation(y_hat, y, regions, params, mode):
    """
    :params y_hat(tensor): prediction
    :params y(tensor): ground truth
    :params params(dict): dict of parameters
    :params mode(str): running mode, such as train, valid or test
    """
    for i, reg in enumerate(regions):
        if params['thresholds']['combine']:
            thre = params['thresholds']['combine_threshold']
        else:
            if reg in params['thresholds'].keys():
                thre = params['thresholds'].get(reg)
            else:
                raise ValueError(f"{reg} not in {params['thresholds'].keys()}")

        sub_yhat = y_hat[i]
        idx_gt0 = sub_yhat >= thre
        sub_yhat[idx_gt0] = 1
        sub_yhat[~idx_gt0] = 0
        y_hat[i] = sub_yhat

    return y_hat

