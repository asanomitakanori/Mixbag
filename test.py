import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix


def test_net(args, 
             epoch,
             model,
             test_loader
             ):
            
    ################## test ###################
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, leave=False):
            data = data.to(args.device)
            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

    gt, pred = np.array(gt), np.array(pred)
    test_acc = (gt == pred).mean()

    test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')

    return test_acc, test_cm