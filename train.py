import logging
from tqdm import tqdm

import numpy as np
import torch

from utils.utils import *
from utils.losses import *

from sklearn.metrics import confusion_matrix


class Run(object):
    """Base class for training, validation, test."""

    def __init__(self, args):
        self.model = model_import(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.train_loader, self.val_loader, self.test_loader = load_data_bags(args)
        # Proportion loss with confidence interval
        self.loss_train, self.loss_val = ProportionLoss_CI(), ProportionLoss()

        self.train_check = True
        self.val_loss = None
        self.cnt = 0
        self.best_val_loss = float("inf")
        self.break_flag = False
        self.best_path = None
        self.output_path = args.output_path
        self.patience = args.patience

    def train(self, args, epoch):
        """Training
        Args:
            args (argparse): contain parameters
            epoch (int): current epoch
        Returns:
            None
        """
        self.model.train()
        losses = []
        if self.train_check == True:
            gt, pred = [], []

        for batch in tqdm(self.train_loader, leave=False):
            (b, n, c, w, h) = batch["img"].size()
            img = batch["img"].reshape(-1, c, w, h).to(args.device)
            label = batch["label"].reshape(-1)
            img, gt_lp = img.to(args.device), batch["label_prop"].to(args.device)

            output = self.model(img)

            if self.train_check == True:
                gt.extend(label.cpu().detach().numpy())
                pred.extend(output.argmax(1).cpu().detach().numpy())

            output = F.softmax(output, dim=1)
            output = output.reshape(b, n, -1)
            pred_lp = output.mean(dim=1)

            loss = self.loss_train(
                pred_lp,
                gt_lp,
                batch["ci_min_value"].to(args.device),
                batch["ci_max_value"].to(args.device),
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())

        train_loss = np.array(losses).mean()
        if self.train_check == True:
            gt, pred = np.array(gt), np.array(pred)
            train_acc = (gt == pred).mean()
        else:
            train_acc = 0

        logging.info(
            "[Epoch: %d/%d] train loss: %.4f, acc: %.4f"
            % (epoch + 1, args.epochs, train_loss, train_acc)
        )

    def val(self, args, epoch):
        """Evaluation
        Args:
            args (argparse): contain parameters
            epoch (int): current epoch

        Returns:
            None
        """
        self.model.eval()
        losses = []
        gt, pred = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                (b, n, c, w, h) = batch["img"].size()
                img = batch["img"].reshape(-1, c, w, h)
                label = batch["label"].reshape(-1)
                img, lp_gt = img.to(args.device), batch["label_prop"].to(args.device)

                output = self.model(img)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(output.argmax(1).cpu().detach().numpy())

                output = F.softmax(output, dim=1)
                output = output.reshape(b, n, -1)
                lp_pred = output.mean(dim=1)

                loss = self.loss_val(lp_pred, lp_gt)
                losses.append(loss.item())

        self.val_loss = np.array(losses).mean()
        gt, pred = np.array(gt), np.array(pred)
        val_acc = (gt == pred).mean()

        logging.info(
            "[Epoch: %d/%d] val loss: %.4f, acc: %.4f"
            % (epoch + 1, args.epochs, self.val_loss, val_acc)
        )

    def early_stopping(self, args, epoch):
        """Early Stopping
        Args:
            args (argparse): contain parameters
            epoch (int): current epoch

        Returns:
            break_flag (True or False): when break_flag is set "True", we stop training. when "False", continue training.
        """
        if self.val_loss > self.best_val_loss:
            self.cnt += 1
            if self.patience == self.cnt:
                self.break_flag = True
        else:
            self.best_val_loss = self.val_loss
            self.cnt = 0
            self.best_path = self.output_path + "/" + str(args.fold) + f"/Best_CP.pkl"
            torch.save(self.model.state_dict(), self.best_path)
        return self.break_flag

    def test(self, args, epoch):
        """Test
        Args:
            args (argparse): contain parameters
            epoch (int): current epoch
            model (torch.tensor): ResNet18
            test_loader (torch.utils.data): test dataloader

        Returns:
            test_acc (float): test accuracy
            test_cm (matrix): confusion matrix
        """

        # Load Best Parameters
        logging.info(f"Model loaded from {self.best_path}")
        self.model.load_state_dict(torch.load(self.best_path, map_location=args.device))

        self.model.eval()
        gt, pred = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, leave=False):
                img = batch["img"].to(args.device)
                output = self.model(img)

                gt.extend(batch["label"].cpu().detach().numpy())
                pred.extend(output.argmax(1).cpu().detach().numpy())

        gt, pred = np.array(gt), np.array(pred)
        test_acc = (gt == pred).mean()
        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize="true")

        logging.info(
            "[Epoch: %d/%d] test acc: %.4f" % (epoch + 1, args.epochs, test_acc)
        )
        logging.info("===============================")

        save_confusion_matrix(
            cm=test_cm,
            path=self.output_path + "/" + str(args.fold) + "/Confusion_matrix.png",
            title="test: acc: %.4f" % test_acc,
        )
