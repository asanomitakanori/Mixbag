
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging

from utils import cal_mIoU, save_confusion_matrix, fix_seed


def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function_train, loss_function_val):
    fix_seed(args.seed)

    train_acc, val_acc, test_acc = [], [], []
    train_mIoU, val_mIoU, test_mIoU = [], [], []
    train_loss, val_loss, test_loss = [], [], []

    train_nega_acc, train_posi_acc = [], []
    val_nega_acc, val_posi_acc = [], []
    test_nega_acc, test_posi_acc = [], []

    best_val_loss = float('inf')
    cnt = 0
    for epoch in range(args.num_epochs):

        ############ train ###################
        s_time = time()
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
            loss = loss_function_train(pred_prop, 
                                       lp, 
                                       min_point.to(args.device), 
                                       max_point.to(args.device)
                                       )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        train_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        train_acc.append((gt == pred).mean())
        train_nega_acc.append((gt == pred)[gt == 0].mean())
        train_posi_acc.append((gt == pred)[gt != 0].mean())

        train_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        train_mIoU.append(cal_mIoU(train_cm))

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, acc: %.4f, mIoU: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time,
                      train_loss[-1], train_acc[-1],
                      train_mIoU[-1]))

        ################# validation ####################
        s_time = time()
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

        val_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        val_acc.append((gt == pred).mean())
        val_nega_acc.append((gt == pred)[gt == 0].mean())
        val_posi_acc.append((gt == pred)[gt != 0].mean())

        val_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        val_mIoU.append(cal_mIoU(val_cm))

        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, acc: %.4f, mIoU: %.4f' %
                        (epoch+1, args.num_epochs, e_time-s_time,
                        val_loss[-1], val_acc[-1],
                        val_mIoU[-1]))


        ################## test ###################
        s_time = time()
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data = data.to(args.device)
                y = model(data)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())

        gt, pred = np.array(gt), np.array(pred)
        test_acc.append((gt == pred).mean())
        test_nega_acc.append((gt == pred)[gt == 0].mean())
        test_posi_acc.append((gt == pred)[gt != 0].mean())

        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        test_mIoU.append(cal_mIoU(test_cm))

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] test acc: %.4f, mIoU: %.4f' %
                        (epoch+1, args.num_epochs, e_time-s_time,
                        test_acc[-1], test_mIoU[-1]))
        logging.info('===============================')

        if best_val_loss > val_loss[-1]:
            best_val_loss = val_loss[-1]
            cnt = 0
            best_epoch = epoch

            torch.save(model.state_dict(), args.output_path  + '-best_model.pkl')

            # save_confusion_matrix(cm=train_cm, path=args.output_path +'unlabel_num' +  str(args.unlabel_num) + '-' + 'aug-' + str(args.augmentation) + '-cm_train.png',
            #                         title='train: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, train_acc[epoch], train_mIoU[epoch]))
            save_confusion_matrix(cm=test_cm, path=args.output_path  + '-cm_test.png',
                                    title='test: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, test_acc[epoch], test_mIoU[epoch]))
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('best epoch: %d, acc: %.4f, test nega: %.4f, posi: %.4f, mIoU: %.4f' %
                        (best_epoch+1, test_acc[best_epoch], test_nega_acc[best_epoch], test_posi_acc[best_epoch], test_mIoU[best_epoch]))
    return train_loss, val_loss, test_acc[best_epoch]