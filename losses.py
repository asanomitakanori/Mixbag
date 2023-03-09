import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input+eps)
    return loss

    
class ProportionLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = cross_entropy_loss(input, target, eps=self.eps) 
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


class ProportionLoss_CI(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, min_area, max_area):
        mask = torch.where((pred<=max_area) & (pred>=min_area), target, pred)
        loss = cross_entropy_loss(mask, target, eps=self.eps) 
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss