import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F


def eval_net(args, 
             epoch,
             model,
             val_loader, 
             loss_function_val):
    """Evaluation without the densecrf with the proportion loss"""

    model.eval()
    losses = []
    gt, pred = [], []
    with torch.no_grad():
        for iteration, (data, label, lp) in enumerate(tqdm(val_loader, leave=False)):
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
            loss = loss_function_val(pred_prop, lp)

            losses.append(loss.item())

    val_loss = np.array(losses).mean()

    gt, pred = np.array(gt), np.array(pred)
    val_acc = (gt == pred).mean()

    logging.info('[Epoch: %d/%d] val loss: %.4f, acc: %.4f' %
                    (epoch+1, args.epochs,
                    val_loss, val_acc))

    return val_loss, val_acc