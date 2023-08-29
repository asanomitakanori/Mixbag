import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F


def train_net(args, model, train_loader, epoch, optimizer, criterion_train):
    """Training: the proportion loss with confidential interval
    Args:
        args (argparse): contain parameters
        train_loader (torch.utils.data): train dataloader
        model (torch.tensor): ResNet18
        epoch (int): current epoch
        optimizer (torch.optim): optimizer such as Adam
        criterion_train: loss function for training

    Returns:
        train_loss (float): average of train loss
        train_acc (float): train accuracy
    """

    model.train()
    losses = []
    gt, pred = [], []
    for iteration, (data, label, lp, min_point, max_point) in enumerate(
        tqdm(train_loader, leave=False)
    ):
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

        loss = criterion_train(
            pred_prop, lp, min_point.to(args.device), max_point.to(args.device)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    train_loss = np.array(losses).mean()
    gt, pred = np.array(gt), np.array(pred)
    train_acc = (gt == pred).mean()

    logging.info(
        f"[Epoch: {epoch+1}/{args.epochs}] train loss: {np.round(train_loss, 4)}, acc: {np.round(train_acc, 4)}"
    )

    return train_loss, train_acc
