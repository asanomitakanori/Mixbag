import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F


def train_net(args, 
             model,
             train_loader,
             optimizer, 
             criterion_train):
    """Training: the proportion loss with confidential interval"""
    model.train()
    losses = []
    gt, pred = [], []
    for iteration, (data, label, lp, min_point, max_point) in enumerate(tqdm(train_loader, leave=False)):
        (b, n, c, w, h) = data.size()
        data = data.reshape(-1, c, w, h)
        label = label.reshape(-1)
        data, lp = data.to(args.device), lp.to(args.device)

        y = model(data)

        gt.extend(label.cpu().detach().numpy())
        pred.extend(y.argmax(1).cpu().detach().numpy())

        confidence = F.softmax(y, dim=1)
        confidence = confidence.reshape(b, n, -1)
        pred_prop = confidence.mean(dim=1)

        loss = criterion_train(pred_prop, 
                                lp, 
                                min_point.to(args.device), 
                                max_point.to(args.device)
                                )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    train_loss = np.array(losses).mean()
    gt, pred = np.array(gt), np.array(pred)
    train_acc = (gt == pred).mean()

    return train_loss, train_acc    