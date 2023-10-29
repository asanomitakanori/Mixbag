import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.dataset import load_data
from utils.losses import (
    PiModelLoss,
    ProportionLoss,
    ProportionLoss_CI,
    VATLoss,
    consistency_loss_function,
)
from utils.utils import calculate_prop, model_import, save_confusion_matrix


class Run(object):
    """Class for training, validation and test."""

    def __init__(self, args):
        self.model = model_import(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # dataloader
        self.train_loader = load_data(args, stage="train")
        self.val_loader = load_data(args, stage="val")
        self.test_loader = load_data(args, stage="test")
        # early stopping parameters
        self.val_loss = None
        self.cnt = 0
        self.best_val_loss = float("inf")
        self.break_flag = False
        self.best_path = None
        self.fold = args.fold
        self.output_path = args.output_path
        self.patience = args.patience
        # Proportion loss with confidence interval
        self.loss_train, self.loss_val = ProportionLoss_CI(), ProportionLoss()

        # Consistency loss
        if args.consistency == "none":
            self.consistency_criterion = None
        elif args.consistency == "vat":
            self.consistency_criterion = VATLoss()
        elif args.consistency == "pi":
            self.consistency_criterion = PiModelLoss()
        else:
            raise NameError("Unknown consistency criterion")

    def train(self, args, epoch):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, leave=False):
            # nb: the number of bags, bs: bag size, c: channel, w: width, h: height
            nb, bs, c, w, h = batch["img"].size()
            img = batch["img"].reshape(-1, c, w, h).to(args.device)
            lp_gt = batch["label_prop"].to(args.device)

            # Consistency loss
            consistency_loss = consistency_loss_function(
                args,
                self.consistency_criterion,
                self.model,
                img,
                self.train_loader,
                epoch,
            )

            output = self.model(img)
            lp_pred = calculate_prop(output, nb, bs)

            loss = self.loss_train(
                lp_pred,
                lp_gt,
                batch["ci_min_value"].to(args.device),
                batch["ci_max_value"].to(args.device),
            )
            loss += consistency_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())

        train_loss = np.array(losses).mean()
        print("[Epoch: %d/%d] train loss: %.4f" % (epoch + 1, args.epochs, train_loss))

    def val(self, args, epoch: int):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                # nb: the number of bags, bs: bag size, c: channel, w: width, h: height
                (nb, bs, c, w, h) = batch["img"].size()
                img = batch["img"].reshape(-1, c, w, h)
                img, lp_gt = img.to(args.device), batch["label_prop"].to(args.device)

                output = self.model(img)
                lp_pred = calculate_prop(output, nb, bs)

                loss = self.loss_val(lp_pred, lp_gt)
                losses.append(loss.item())

        self.val_loss = np.array(losses).mean()
        print("[Epoch: %d/%d] val loss: %.4f" % (epoch + 1, args.epochs, self.val_loss))

    def early_stopping(self, args, epoch: int):
        """Early Stopping
        Args:
            args (argparse): contain parameters
            epoch (int): current epoch

        Returns:
            break_flag (True or False): when break_flag is "True", stop training. when it is "False", continue training.
        """
        if self.val_loss > self.best_val_loss:
            self.cnt += 1
            if self.patience == self.cnt:
                self.break_flag = True
        else:
            self.best_val_loss = self.val_loss
            self.cnt = 0
            self.best_path = self.output_path + "/" + str(self.fold) + "/Best_CP.pkl"
            torch.save(self.model.state_dict(), self.best_path)
        return self.break_flag

    def test(self, args):
        self.model.load_state_dict(torch.load(self.best_path, map_location=args.device))
        self.model.eval()
        gt, pred = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, leave=False):
                img = batch["img"].to(args.device)
                assert len(img.shape) in [4, 5], "img should be 4 or 5 dims"

                output = self.model(img)

                gt.extend(batch["label"].cpu().detach().numpy())
                pred.extend(output.argmax(1).cpu().detach().numpy())

        gt, pred = np.array(gt), np.array(pred)
        test_acc = (gt == pred).mean()

        # calculate confusion matrix and save confusion matrix
        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize="true")
        save_confusion_matrix(
            cm=test_cm,
            path=self.output_path + "/" + str(self.fold) + "/Confusion_matrix.png",
            title="test: acc: %.4f" % test_acc,
        )
        print(f"Test acc: {np.round(test_acc, 4)}")
        print("========================")
