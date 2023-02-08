import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from time import time
from tqdm import tqdm
import logging

from utils import Dataset, DatasetBag, cal_mIoU, fix_seed, make_folder, save_confusion_matrix



def main(args):

    fix_seed(args.seed)

    make_folder(args.output_path)
    args.output_path += '%s/' % (args.dataset)
    make_folder(args.output_path)
    args.output_path += 'SL/'
    make_folder(args.output_path)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)

    ######### load data #######
    train_data = np.load('data/%s/train_data.npy'%(args.dataset))
    train_label = np.load('data/%s/train_label.npy'%(args.dataset))
    test_data = np.load('data/%s/test_data.npy'%(args.dataset))
    test_label = np.load('data/%s/test_label.npy'%(args.dataset))
   
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=args.split_ratio, random_state=42)

    train_dataset = Dataset(data=train_data, label=train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers)
    val_dataset = Dataset(data=val_data, label=val_label)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)
    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)

    fix_seed(args.seed)
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, max(test_label)+1)
    model = model.to(args.device)

    weight = 1 / np.eye(max(test_label)+1)[train_label].sum(axis=0)
    weight /= weight.sum()
    weight = torch.tensor(weight).float().to(args.device)
    loss_function = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_acc, val_acc, test_acc = [], [], []
    train_mIoU, val_mIoU, test_mIoU = [], [], []
    
    train_nega_acc, train_posi_acc = [], []
    val_nega_acc, val_posi_acc = [], []
    test_nega_acc, test_posi_acc = [], []

    best_val_mIoU = 0
    for epoch in range(args.num_epochs):

        ############ train ###################
        s_time = time()
        model.train()
        gt, pred = [], []
        for data, label in tqdm(train_loader, leave=False):
            data, label = data.to(args.device), label.to(args.device)
            y = model(data)
            loss = loss_function(y, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

        gt, pred = np.array(gt), np.array(pred)
        train_acc.append((gt==pred).mean())
        train_nega_acc.append((gt==pred)[gt==0].mean())
        train_posi_acc.append((gt==pred)[gt!=0].mean())

        train_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        train_mIoU.append(cal_mIoU(train_cm))
        
        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] acc: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, 
                 train_acc[-1],train_mIoU[-1], train_nega_acc[-1], train_posi_acc[-1]))

        ################## test ###################
        s_time = time()
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(val_loader, leave=False):
                data = data.to(args.device)
                y = model(data)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())


        gt, pred = np.array(gt), np.array(pred)
        val_acc.append((gt==pred).mean())
        val_nega_acc.append((gt==pred)[gt==0].mean())
        val_posi_acc.append((gt==pred)[gt!=0].mean())

        val_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        val_mIoU.append(cal_mIoU(val_cm))

        logging.info('[Epoch: %d/%d (%ds)] val acc: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, 
                 val_acc[-1], val_mIoU[-1], val_nega_acc[-1], val_posi_acc[-1]))

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
        test_acc.append((gt==pred).mean())
        test_nega_acc.append((gt==pred)[gt==0].mean())
        test_posi_acc.append((gt==pred)[gt!=0].mean())

        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        test_mIoU.append(cal_mIoU(test_cm))

        e_time = time()
        
        logging.info('[Epoch: %d/%d (%ds)] test acc: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, 
                 test_acc[-1], test_mIoU[-1], test_nega_acc[-1], test_posi_acc[-1]))
        logging.info('===============================')

        if best_val_mIoU < val_mIoU[-1]:
            best_val_mIoU = val_mIoU[-1]
            cnt = 0
            best_epoch = epoch

            torch.save(model.state_dict(), args.output_path + 'best_model.pkl')

            save_confusion_matrix(cm=train_cm, path=args.output_path+'cm_train.png',
                                  title='train: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, train_acc[epoch], train_mIoU[epoch]))
            save_confusion_matrix(cm=test_cm, path=args.output_path+'cm_test.png',
                                  title='test: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, test_acc[epoch], test_mIoU[epoch]))
        else:
            cnt += 1
            if args.patience==cnt:
                break
    
    logging.info('best epoch: %d, acc: %.4f, test nega: %.4f, posi: %.4f, mIoU: %.4f' %
          (best_epoch+1, test_acc[best_epoch], test_nega_acc[best_epoch], test_posi_acc[best_epoch], test_mIoU[best_epoch]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,
                        type=int, help='seed value')
    parser.add_argument('--dataset', default='mnist',
                        type=str, help='name of dataset')
    parser.add_argument('--device', default='cuda:0',
                        type=str, help='device')
    parser.add_argument('--split_ratio', default=0.25,
                        type=float, help='split ratio')
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batch size for training.')
    parser.add_argument('--num_epochs', default=1000, type=int,
                        help='number of epochs for training.')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of workers for training.')
    parser.add_argument('--patience', default=10,
                        type=int, help='patience of early stopping')
    parser.add_argument('--lr', default=3e-4,
                        type=float, help='learning rate')
    parser.add_argument('--output_path',
                        default='result/', type=str, help="output file name")
    args = parser.parse_args()

    #################
    main(args)