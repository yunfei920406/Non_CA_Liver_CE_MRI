# -*- coding: utf-8 -*-

"""
@author: Yunfei Zhang
@software: PyCharm
@file: Loss_Functions.py
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def L2_Loss(x,y):
   return F.mse_loss(x,y)



def L1_Loss(x,y):
    return torch.abs(x-y).mean()





def D_Loss_real(img_real_logit):
    return F.relu(1-img_real_logit).mean()


def D_Loss_fake(img_fake_logit):
    return F.relu(1+img_fake_logit).mean()

def G_Loss_real(img_fake_logit):
    return F.relu(1-img_fake_logit).mean()


a = torch.rand(2,2)
torch.exp(a)/(torch.exp(a).sum())

def global_softmax(tensor):
    sum_ = torch.exp(tensor).sum()
    return torch.exp(tensor)/sum_



def Identity_Loss(img_recon,img_real):
    """
    :param img_recon: img_recon is img_real ===> A ====> img_recon
    :param img_real: img_real
    :return: The Identity Loss,L1 loss
    """
    return torch.abs(img_recon-img_real).mean()


def Loss_real_BCE(y_logit,GPU = False):
    y_true = torch.ones_like(y_logit)
    return torch.binary_cross_entropy_with_logits(y_logit,y_true).mean()


def Loss_fake_BCE(y_logit,GPU = False):
    y_true = torch.zeros_like(y_logit)
    return torch.binary_cross_entropy_with_logits(y_logit,y_true).mean()


