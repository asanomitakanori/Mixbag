import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input+eps)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric="ce", eps=1e-8):
        super().__init__()
        self.metric = metric
        self.eps = eps

    def forward(self, input, target):
        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.mean(loss, dim=-1)
        return loss

class PartialProportionLoss(nn.Module):
    def __init__(self, w_n=1, w_p=1, metric="ce", eps=1e-8):
        super().__init__()
        self.w_n = w_n
        self.w_p = w_p
        self.metric = metric
        self.eps = eps
        self.posi_loss = ProportionLoss(metric, eps)
        self.nega_loss = ProportionLoss(metric, eps)

    def forward(self, input, target):
        posi_nega = (target[:, 0]!=1).long()
        
        loss_nega = (1-posi_nega) * self.nega_loss(input, target)
        
        input = input[:, 1:] / input[:, 1:].sum(dim=1, keepdims=True).repeat(1, input.size(-1)-1)
        loss_posi = posi_nega * self.posi_loss(input, target[:, 1:])

        loss = self.w_n * loss_nega + self.w_p * loss_posi
        
        loss = loss.mean()
    
        return loss, loss_nega[posi_nega==0], loss_posi[posi_nega==1]
