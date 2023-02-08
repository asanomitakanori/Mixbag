import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from time import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from model import Attention
from losses import PiModelLoss, ProportionLoss, VATLoss, get_rampup_weight
from eval_MIL_LPLP import evaluation_MIL_LLP
from utils import cal_OP_PC_mIoU, cal_mIoU, fix_seed, make_folder, save_confusion_matrix


def main(args):
    fix_seed(args.seed)
    if args.dataset == 'wsi':
        from utils_WSI import load_data, load_data_MIL
    else:
        from utils_toy import load_data, load_data_MIL

    ############ create folder ############
    make_folder(args.output_path)
    args.output_path += '%s/' % (args.dataset)
    make_folder(args.output_path)

    args.output_path += 'kFold_%s/' % (args.kFold)
    make_folder(args.output_path)

    ############ create loger ############
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)

    ############ create loader ############
    if args.dataset == 'wsi':
        if args.is_MIL:
            train_loader, val_loader, test_loader = load_data_MIL(args)
        else:
            train_loader, val_loader, test_loader = load_data(
                args, is_full_proportion=True)
    else:
        if args.is_MIL:
            train_loader, val_loader, test_loader = load_data_MIL(args)
        else:
            train_loader, val_loader, test_loader = load_data(
                args, is_full_proportion=True)
    logging.info(next(iter(train_loader))[2])

    ############ define mdoel ############
    if args.is_MIL:
        args.num_classes -= 1

    if args.is_pretrain:
        model = resnet18(weights='DEFAULT')
    else:
        model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(args.device)

    ############ define loss function & optimizer ############
    loss_function = ProportionLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.consistency == 'none':
        consistency_criterion = None
    elif args.consistency == 'vat':
        consistency_criterion = VATLoss()
    elif args.consistency == 'pi':
        consistency_criterion = PiModelLoss()
    else:
        raise NameError('Unknown consistency criterion')


    ############ training ############
    fix_seed(args.seed)

    if args.dataset == 'wsi':
        if args.is_MIL:
            training = TrainWSI_different_bs()
        else:
            training = TrainWSI()
    else:
        if args.is_MIL:
            training = TrainToy_different_bs()
        else:
            training = TrainToy()

    for epoch in range(args.num_epochs):
        logging.info('loss weight: %.4f, %.4f' % (args.w_n, args.w_p))

        ############ train ############
        message = training.train(
            model, train_loader, loss_function, consistency_criterion, optimizer, args)
        logging.info('[Epoch: %d/%d]%s' % (epoch+1, args.num_epochs, message))

        ############ validation ############
        message = training.validation(model, val_loader, loss_function, args)
        logging.info('[Epoch: %d/%d]%s' % (epoch+1, args.num_epochs, message))

        ############ save ############
        training.save(args.output_path)
        training.vis_training_curve(args.output_path)

        if training.early_stopping(training.val_loss[-1], model):
            break

        logging.info('------------------------------------------------------')

    ############ test ############
    if args.is_MIL == 0:
        model.load_state_dict(torch.load(args.output_path+'best_model_PL.pkl'))
        message = training.test(model, test_loader, loss_function, args)
        training.save(args.output_path)
        save_confusion_matrix(cm=training.test_cm, path=args.output_path+'cm_test.png',
                              title='test: epoch: %d, acc: %.4f, mIoU: %.4f' % (
                                  training.best_epoch+1, training.test_acc[-1], training.test_mIoU[-1]))
        logging.info('[best epoch: %d]%s' % (training.best_epoch+1, message))
    else:
        model.load_state_dict(torch.load(args.output_path+'best_model_PL.pkl'))
        model_LLP = model

        model_MIL = Attention(args.num_classes+1).to(args.device)
        model_MIL.load_state_dict(torch.load(
            args.output_path+'best_model_PPL+MIL.pkl'))

        acc, PC, mIoU = evaluation_MIL_LLP(model_MIL, model_LLP, test_loader, args)
        training.test_acc.append(acc)
        training.test_PC.append(PC)
        training.test_mIoU.append(mIoU)
        training.save(args.output_path)
        logging.info('[Naive] acc: %.4f, PC: %.4f, mIoU: %.4f' % (acc, PC, mIoU))


class Train():
    def __init__(self):
        self.train_acc, self.val_acc, self.test_acc = [], [], []
        self.train_PC, self.val_PC, self.test_PC = [], [], []
        self.train_mIoU, self.val_mIoU, self.test_mIoU = [], [], []
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.train_nega_acc, self.train_posi_acc = [], []
        self.val_nega_acc, self.val_posi_acc = [], []
        self.test_nega_acc, self.test_posi_acc = [], []

        self.train_cm, self.val_cm, self.test_cm = None, None, None

        self.early_stopping_criterion = float('inf')
        self.early_stopping_cnt = 0
        self.epoch = 0
        self.best_epoch = 0

    def save(self, path):
        self.log = {}
        self.log['train_acc'], self.log['val_acc'], self.log['test_acc'] = self.train_acc, self.val_acc, self.test_acc
        self.log['train_PC'], self.log['val_PC'], self.log['test_PC'] = self.train_PC, self.val_PC, self.test_PC
        self.log['train_mIoU'], self.log['val_mIoU'], self.log['test_mIoU'] = self.train_mIoU, self.val_mIoU, self.test_mIoU
        self.log['train_loss'], self.log['val_loss'], self.log['test_loss'] = self.train_loss, self.val_loss, self.test_loss
        self.log['train_nega_acc'], self.log['train_posi_acc'] = self.train_nega_acc, self.train_posi_acc
        self.log['val_nega_acc'], self.log['val_posi_acc'] = self.val_nega_acc, self.val_posi_acc
        self.log['test_nega_acc'], self.log['test_posi_acc'] = self.test_nega_acc, self.test_posi_acc

        self.log['train_cm'], self.log['val_cm'], self.log['test_cm'] = self.train_cm, self.val_cm, self.test_cm
        self.log['best_epoch'] = self.best_epoch

        np.save(path+'training_log_PL', self.log)

    def vis_training_curve(self, path):
        # np.load(path+'training_log.npy', allow_pickle=True).item()
        plt.plot(self.train_mIoU, label='train')
        plt.plot(self.val_mIoU, label='val')
        plt.plot(self.test_mIoU, label='test')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(path+'curve_acc.png')
        plt.close()

    def early_stopping(self, value, model):
        self.epoch += 1
        if self.early_stopping_criterion > value:
            self.early_stopping_cnt = 0
            self.early_stopping_criterion = value
            self.best_epoch = self.epoch - 1

            torch.save(model.state_dict(),
                       args.output_path + 'best_model_PL.pkl')

            save_confusion_matrix(cm=self.train_cm, path=args.output_path+'cm_train.png',
                                  title='train: epoch: %d, acc: %.4f, mIoU: %.4f' % (
                                        self.best_epoch+1, self.train_acc[self.best_epoch], self.train_mIoU[self.best_epoch]))
            save_confusion_matrix(cm=self.val_cm, path=args.output_path+'cm_val.png',
                                  title='val: epoch: %d, acc: %.4f, mIoU: %.4f' % (
                                      self.best_epoch+1, self.val_acc[self.best_epoch], self.val_mIoU[self.best_epoch]))

        else:
            self.early_stopping_cnt += 1
            if (args.patience == self.early_stopping_cnt) and (args.is_early_stopping):
                return 1

        return 0


class TrainToy(Train):
    def __init__(self):
        super().__init__()

    def train(self, model, loader, loss_function, consistency_criterion, optimizer, args):
        s_time = time()
        model.train()
        losses = []
        gt, pred = [], []
        for iteration, batch in enumerate(tqdm(loader, leave=False)):
            data, label, lp = batch[0], batch[1], batch[2]
            bag_label = (lp[:, 0] != 1).long().to(args.device)

            (b, n, c, w, h) = data.size()
            data = data.reshape(-1, c, w, h)
            label = label.reshape(-1)
            data, lp = data.to(args.device), lp.to(args.device)

            # consistency
            if args.consistency != 'none':
                consistency_loss = consistency_criterion(model, data)
                consistency_rampup = 0.4 * args.num_epochs * len(loader)
                alpha = get_rampup_weight(0.05, self.epoch, consistency_rampup)
                consistency_loss = alpha * consistency_loss
            else:
                consistency_loss = torch.tensor(0.)
            
            # main
            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(b, n, -1)
            pred_prop = confidence.mean(dim=1)
            loss = loss_function(pred_prop, lp)
            loss_nega = (bag_label == 0)*loss
            loss_posi = (bag_label == 1)*loss
            loss = args.w_n * loss_nega + args.w_p * loss_posi
            loss = loss.mean()

            loss += consistency_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        self.train_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        self.train_acc.append((gt == pred).mean())
        self.train_nega_acc.append((gt == pred)[gt == 0].mean())
        self.train_posi_acc.append((gt == pred)[gt != 0].mean())

        self.train_cm = confusion_matrix(
            y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(self.train_cm)
        self.train_PC.append(PC)
        self.train_mIoU.append(mIoU)

        e_time = time()

        message = '(%ds) train loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.train_loss[-1], self.train_acc[-1], self.train_PC[-1],
            self.train_mIoU[-1], self.train_nega_acc[-1], self.train_posi_acc[-1])

        return message

    def evaluation(self, model, loader, loss_function, args):
        model.eval()
        total_loss = 0
        gt, pred = [], []
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(loader, leave=False)):
                data, label, lp = batch[0], batch[1], batch[2]
                bag_label = (lp[:, 0] != 1).long().to(args.device)

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
                loss = loss_function(pred_prop, lp)
                loss_nega = (bag_label == 0)*loss
                loss_posi = (bag_label == 1)*loss
                loss = args.w_n * loss_nega + args.w_p * loss_posi
                loss = loss.mean()

                total_loss += loss.item()

        total_loss = total_loss / len(loader)

        gt, pred = np.array(gt), np.array(pred)
        acc = (gt == pred).mean()
        nega_acc = (gt == pred)[gt == 0].mean()
        posi_acc = (gt == pred)[gt != 0].mean()

        cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(cm)

        return total_loss, acc, nega_acc, posi_acc, PC, mIoU, cm

    def validation(self, model, loader, loss_function, args):
        s_time = time()

        loss, acc, nega_acc, posi_acc, PC, mIoU, cm = \
            self.evaluation(model, loader, loss_function, args)

        self.val_loss.append(loss)
        self.val_acc.append(acc)
        self.val_nega_acc.append(nega_acc)
        self.val_posi_acc.append(posi_acc)
        self.val_PC.append(PC)
        self.val_mIoU.append(mIoU)
        self.val_cm = cm

        e_time = time()

        message = '(%ds) val loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.val_loss[-1], self.val_acc[-1], self.val_PC[-1],
            self.val_mIoU[-1], self.val_nega_acc[-1], self.val_posi_acc[-1])

        return message

    def test(self, model, loader, loss_function, args):
        s_time = time()

        loss, acc, nega_acc, posi_acc, PC, mIoU, cm = \
            self.evaluation(model, loader, loss_function, args)

        self.test_loss.append(loss)
        self.test_acc.append(acc)
        self.test_nega_acc.append(nega_acc)
        self.test_posi_acc.append(posi_acc)
        self.test_PC.append(PC)
        self.test_mIoU.append(mIoU)
        self.test_cm = cm

        e_time = time()

        message = '(%ds) test loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.test_loss[-1], self.test_acc[-1], self.test_PC[-1],
            self.test_mIoU[-1], self.test_nega_acc[-1], self.test_posi_acc[-1])

        return message


class TrainToy_different_bs(TrainToy):
    def __init__(self):
        super().__init__()

    def train(self, model, loader, loss_function, consistency_criterion, optimizer, args):
        s_time = time()
        model.train()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        gt, pred = [], []
        for iteration, (data, label, proportion) in enumerate(tqdm(loader, leave=False)):
            data = data[0]
            gt.extend(label[0].numpy())
            b_list.append(b_list[-1]+data.size(0))
            mb_data.extend(data)
            mb_proportion.extend(proportion)

            if (iteration+1) % args.mini_batch == 0 or (iteration + 1) == len(loader):
                mb_data = torch.stack(mb_data)
                mb_proportion = torch.stack(mb_proportion)
                mb_data = mb_data.to(args.device)
                mb_proportion = mb_proportion.to(args.device)

                # consistency
                if args.consistency != 'none':
                    consistency_loss = consistency_criterion(model, mb_data)
                    consistency_rampup = 0.4 * args.num_epochs * len(loader) / args.mini_batch
                    alpha = get_rampup_weight(0.05, self.epoch, consistency_rampup)
                    consistency_loss = alpha * consistency_loss
                else:
                    consistency_loss = torch.tensor(0.)
                
                # main
                y = model(mb_data)
                pred.extend(y.argmax(1).cpu().detach().numpy())
                confidence = F.softmax(y, dim=1)
                pred_prop = torch.zeros(mb_proportion.size(
                    0), args.num_classes).to(args.device)
                for i, n in enumerate(range(mb_proportion.size(0))):
                    if b_list[n] != b_list[n+1]:
                        pred_prop[n] = torch.mean(
                            confidence[b_list[n]: b_list[n+1]], dim=0)
                    else:
                        mb_proportion = torch.cat(
                            [mb_proportion[:i], mb_proportion[i+1:]], dim=0)
                loss = loss_function(pred_prop, mb_proportion)
                loss = loss.mean()

                loss += consistency_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                b_list = [0]
                mb_data, mb_proportion = [], []

                losses.append(loss.item())

        self.train_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        self.train_acc.append((gt == pred).mean())
        self.train_nega_acc.append((gt == pred)[gt == 0].mean())
        self.train_posi_acc.append((gt == pred)[gt != 0].mean())

        self.train_cm = confusion_matrix(
            y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(self.train_cm)
        self.train_PC.append(PC)
        self.train_mIoU.append(mIoU)

        e_time = time()

        message = '(%ds) train loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.train_loss[-1], self.train_acc[-1], self.train_PC[-1],
            self.train_mIoU[-1], self.train_nega_acc[-1], self.train_posi_acc[-1])

        return message


class TrainWSI(TrainToy):
    def __init__(self):
        super().__init__()

    def test(self, model, loader, loss_function, args):
        s_time = time()
        loss, acc, PC, mIoU, cm = self.evaluation_wsi(
            model, loader, loss_function, args)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        self.test_PC.append(PC)
        self.test_mIoU.append(mIoU)
        self.test_cm = cm
        e_time = time()
        message = '(%ds) test loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f' % (
            e_time-s_time, self.test_loss[-1], self.test_acc[-1], self.test_PC[-1], self.test_mIoU[-1])
        return message

    def evaluation_wsi(self, model, loader, loss_function, args):
        model.eval()
        gt, pred = [], []
        total_loss = 0
        with torch.no_grad():
            for data, label, proportion, _ in tqdm(loader, leave=False):
                bag_label = (proportion[:, 0] != 1).long().to(args.device)
                data, proportion = data[0], proportion[0]
                gt.extend(label[0])

                confidence = []
                if (data.size(0) % args.batch_size) == 0:
                    J = int((data.size(0)//args.batch_size))
                else:
                    J = int((data.size(0)//args.batch_size)+1)

                for j in range(J):
                    if j+1 != J:
                        data_j = data[j*args.batch_size: (j+1)*args.batch_size]
                    else:
                        data_j = data[j*args.batch_size:]

                    data_j = data_j.to(args.device)
                    y = model(data_j)
                    pred.extend(y.argmax(1).cpu().detach().numpy())
                    confidence.extend(
                        F.softmax(y, dim=1).cpu().detach().numpy())

                pred_prop = torch.tensor(np.array(confidence)).mean(dim=0)
                loss = loss_function(pred_prop, proportion)
                loss_nega = (bag_label == 0)*loss
                loss_posi = (bag_label == 1)*loss
                loss = args.w_n * loss_nega + args.w_p * loss_posi
                loss = loss.mean()

                total_loss += loss.item()

        total_loss = total_loss / len(loader)

        acc = np.array(np.array(gt) == np.array(pred)).mean()
        cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(cm)

        return total_loss, acc, PC, mIoU, cm


class TrainWSI_different_bs(TrainWSI):
    def __init__(self):
        super().__init__()

    def train(self, model, loader, loss_function, consistency_criterion, optimizer, args):
        s_time = time()
        model.train()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        mb_bag_label = []
        gt, pred = [], []
        for iteration, (data, label, proportion, _) in enumerate(tqdm(loader, leave=False)):
            data = data[0]
            gt.extend(label[0].numpy())
            b_list.append(b_list[-1]+data.size(0))
            mb_data.extend(data)
            mb_proportion.extend(proportion)

            bag_label = (proportion[:, 0] != 1).long()
            mb_bag_label.extend(bag_label)

            if (iteration+1) % args.mini_batch == 0 or (iteration + 1) == len(loader):
                mb_data = torch.stack(mb_data)
                mb_proportion = torch.stack(mb_proportion)
                mb_bag_label = torch.stack(mb_bag_label)
                mb_data = mb_data.to(args.device)
                mb_proportion = mb_proportion.to(args.device)
                mb_bag_label = mb_bag_label.to(args.device)

                # consistency
                if args.consistency != 'none':
                    consistency_loss = consistency_criterion(model, mb_data)
                    consistency_rampup = 0.4 * args.num_epochs * len(loader) / args.mini_batch
                    alpha = get_rampup_weight(0.05, self.epoch, consistency_rampup)
                    consistency_loss = alpha * consistency_loss
                else:
                    consistency_loss = torch.tensor(0.)
                
                # main
                y = model(mb_data)
                pred.extend(y.argmax(1).cpu().detach().numpy())
                confidence = F.softmax(y, dim=1)
                pred_prop = torch.zeros(mb_proportion.size(
                    0), args.num_classes).to(args.device)
                for i, n in enumerate(range(mb_proportion.size(0))):
                    if b_list[n] != b_list[n+1]:
                        pred_prop[n] = torch.mean(
                            confidence[b_list[n]: b_list[n+1]], dim=0)
                    else:
                        mb_proportion = torch.cat(
                            [mb_proportion[:i], mb_proportion[i+1:]], dim=0)
                loss = loss_function(pred_prop, mb_proportion)
                if args.is_MIL == 0:
                    loss_nega = (mb_bag_label == 0)*loss
                    loss_posi = (mb_bag_label == 1)*loss
                    loss = args.w_n * loss_nega + args.w_p * loss_posi
                loss = loss.mean()

                loss += consistency_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                b_list = [0]
                mb_data, mb_proportion, mb_bag_label = [], [], []

                losses.append(loss.item())

        self.train_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        self.train_acc.append((gt == pred).mean())
        self.train_nega_acc.append((gt == pred)[gt == 0].mean())
        self.train_posi_acc.append((gt == pred)[gt != 0].mean())

        self.train_cm = confusion_matrix(
            y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(self.train_cm)
        self.train_PC.append(PC)
        self.train_mIoU.append(mIoU)

        e_time = time()

        message = '(%ds) train loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.train_loss[-1], self.train_acc[-1], self.train_PC[-1],
            self.train_mIoU[-1], self.train_nega_acc[-1], self.train_posi_acc[-1])

        return message

    def validation(self, model, loader, loss_function, args):
        s_time = time()
        model.eval()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        mb_bag_label = []
        gt, pred = [], []
        with torch.no_grad():
            for iteration, (data, label, proportion, _) in enumerate(tqdm(loader, leave=False)):
                bag_label = (proportion[:, 0] != 1).long().to(args.device)
                data = data[0]
                gt.extend(label[0].numpy())
                b_list.append(b_list[-1]+data.size(0))
                mb_data.extend(data)
                mb_proportion.extend(proportion)

                bag_label = (proportion[:, 0] != 1).long()
                mb_bag_label.extend(bag_label)

                if (iteration+1) % args.mini_batch == 0 or (iteration + 1) == len(loader):
                    mb_data = torch.stack(mb_data)
                    mb_proportion = torch.stack(mb_proportion)
                    mb_bag_label = torch.stack(mb_bag_label)
                    mb_data = mb_data.to(args.device)
                    mb_proportion = mb_proportion.to(args.device)
                    mb_bag_label = mb_bag_label.to(args.device)

                    y = model(mb_data)
                    pred.extend(y.argmax(1).cpu().detach().numpy())
                    confidence = F.softmax(y, dim=1)
                    pred_prop = torch.zeros(mb_proportion.size(
                        0), args.num_classes).to(args.device)
                    for i, n in enumerate(range(mb_proportion.size(0))):
                        if b_list[n] != b_list[n+1]:
                            pred_prop[n] = torch.mean(
                                confidence[b_list[n]: b_list[n+1]], dim=0)
                        else:
                            mb_proportion = torch.cat(
                                [mb_proportion[:i], mb_proportion[i+1:]], dim=0)
                    loss = loss_function(pred_prop, mb_proportion)
                    if args.is_MIL == 0:
                        loss_nega = (mb_bag_label == 0)*loss
                        loss_posi = (mb_bag_label == 1)*loss
                        loss = args.w_n * loss_nega + args.w_p * loss_posi
                    loss = loss.mean()

                    b_list = [0]
                    mb_data, mb_proportion, mb_bag_label = [], [], []

                    losses.append(loss.item())

        self.val_loss.append(np.array(losses).mean())

        gt, pred = np.array(gt), np.array(pred)
        self.val_acc.append((gt == pred).mean())
        self.val_nega_acc.append((gt == pred)[gt == 0].mean())
        self.val_posi_acc.append((gt == pred)[gt != 0].mean())

        self.val_cm = confusion_matrix(
            y_true=gt, y_pred=pred, normalize='true')
        _, PC, mIoU = cal_OP_PC_mIoU(self.val_cm)
        self.val_PC.append(PC)
        self.val_mIoU.append(mIoU)

        e_time = time()

        message = '(%ds) val loss: %.4f, acc: %.4f, PC: %.4f, mIoU: %.4f, nega: %.4f, posi: %.4f' % (
            e_time-s_time, self.val_loss[-1], self.val_acc[-1], self.val_PC[-1],
            self.val_mIoU[-1], self.val_nega_acc[-1], self.val_posi_acc[-1])

        return message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='seed value')
    parser.add_argument('--dataset', default='cifar10',
                        type=str, help='name of dataset')
    parser.add_argument('--num_classes', default=10,
                        type=int, help='number of classes')
    parser.add_argument('--kFold', default=0, type=int,
                        help='k-fold closs validation')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--split_ratio', default=0.25,
                        type=float, help='split ratio')
    parser.add_argument('--is_pretrain', default=1,
                        type=float, help='split ratio')
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batch size for training.')
    parser.add_argument('--mini_batch', default=16, type=int,
                        help='mini batch for training.')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of epochs for training.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for training.')
    parser.add_argument('--is_early_stopping', default=1,
                        type=int, help='whether using early stopping')
    parser.add_argument('--patience', default=10, type=int,
                        help='patience of early stopping')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--w_n', default=1, type=float,
                        help='weight of negative proportion loss')
    parser.add_argument('--w_p', default=1, type=float,
                        help='weight of positive proportion loss')
    parser.add_argument('--consistency', default='vat', type=str,
                        help='consistency of proportion loss')
    parser.add_argument('--output_path', default='debug/',
                        type=str, help="output file name")

    parser.add_argument('--num_sampled_instances', default=32,
                        type=int, help="number of the sampled instnace")
    parser.add_argument('--is_MIL', default=0, type=int, help="after MIL")

    args = parser.parse_args()

    #################
    main(args)
