import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
import torch.nn.functional as F

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_exp_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))   

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def TPL_loss(outputs, labels, criterion):
    gt_mask = _get_gt_mask(outputs[:, 0, ...], labels)
    other_mask = _get_other_mask(outputs[:, 0, ...], labels)
    T = outputs.size(1)
    _outputs = outputs.detach()
    Loss_es = 0
    for t in range(T):
        past_time_sum = torch.sum(_outputs[:, 0:t, ...], dim=1)
        pre_time_sum = (past_time_sum + outputs[:, t, ...]) / (t+1)
        pre_mean_out = F.softmax(pre_time_sum, dim=1)
        loss_target = ((pre_mean_out * gt_mask).sum(dim=1)).mean()
        loss_other = ((pre_mean_out * other_mask).sum(dim=1)).mean()
        Loss_es += criterion(outputs[:, t, ...], labels) + (loss_target / loss_other) ** ((T - t) / T)
    Loss_es = Loss_es / T 
    return Loss_es # L_Total

def TET_loss(outputs, labels, criterion):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    return Loss_es # L_Total

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()  # one-hot target
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool() # one-hot target姣忎釜鍏冪礌鍙栧弽
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
