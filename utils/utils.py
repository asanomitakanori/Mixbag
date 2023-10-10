import os
import random
import seaborn as sns

import numpy as np
from scipy.stats import norm
from hydra.utils import to_absolute_path as to_abs_path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet18


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_confusion_matrix(cm, path, title=""):
    sns.heatmap(cm, annot=True, cmap="Blues_r", fmt=".4f")
    plt.xlabel("pred")
    plt.ylabel("GT")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def load_data_bags(args):  # Toy
    ######### load data #######
    test_data = np.load(to_abs_path("data/%s/test_data.npy" % (args.dataset)))
    test_label = np.load(to_abs_path("data/%s/test_label.npy" % (args.dataset)))

    train_bags = np.load(
        to_abs_path("data/%s/%d/train_bags.npy" % (args.dataset, args.fold))
    )
    train_labels = np.load(
        to_abs_path("data/%s/%d/train_labels.npy" % (args.dataset, args.fold))
    )
    train_lps = np.load(
        to_abs_path("data/%s/%d/train_original_lps.npy" % (args.dataset, args.fold))
    )
    val_bags = np.load(
        to_abs_path("data/%s/%d/val_bags.npy" % (args.dataset, args.fold))
    )
    val_labels = np.load(
        to_abs_path("data/%s/%d/val_labels.npy" % (args.dataset, args.fold))
    )
    val_lps = np.load(
        to_abs_path("data/%s/%d/val_original_lps.npy" % (args.dataset, args.fold))
    )

    train_dataset = Dataset_Mixbag(
        args=args,
        data=train_bags,
        label=train_labels,
        lp=train_lps,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mini_batch,
        worker_init_fn=worker_init_fn(args.seed),
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = Dataset_Val(
        data=val_bags,
        label=val_labels,
        lp=val_lps,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.mini_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataset = Dataset_Test(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


class Dataset_Base(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(self.data)
        if len(self.data[0].shape) == 4:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5071), (0.2673))]
            )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Overload this function in your Dataset."""
        raise NotImplementedError


class Dataset_Val(Dataset_Base):
    def __init__(self, data, label, lp):
        super().__init__(data, label)
        self.lp = lp

    def __getitem__(self, idx):
        data, label, lp = self.data[idx], self.label[idx], self.lp[idx]
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
        # bs: bag size, w: width, h: height, c: channel
        (bs, w, h, c) = data.shape
        trans_data = torch.zeros((bs, c, w, h))
        for i in range(bs):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        return {
            "img": data,
            "label": torch.tensor(label).long(),
            "label_prop": torch.tensor(lp).float(),
        }


class Dataset_Test(Dataset_Base):
    def __init__(self, data, label):
        super().__init__(data, label)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        if len(img.shape) != 3:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        img = self.transform(img)

        return {"img": img, "label": torch.tensor(label).long()}


class Dataset_Mixbag(Dataset_Base):
    def __init__(self, args, data, label, lp):
        super().__init__(data, label)
        fix_seed(args.seed)
        self.lp = lp
        self.classes = args.classes
        self.choice = args.choice
        self.CI = args.confidence_interval

    def sampling(self, index_i: list, index_j: list, lp_i: float, lp_j: float):
        """Sampling methods
        Args:
            index_i (list):
            index_j (list):
            sampled_lp (float)
        Returns:
            expected_lp (float):
            index_i (list):
            index_j (list):
            min_error (float):
            max_error (float):
        """
        if self.choice == "half":
            index_i, index_j = (
                index_i[0 : index_i.shape[0] // 2],
                index_j[0 : index_j.shape[0] // 2],
            )

        elif self.choice == "uniform":
            sep = np.random.randint(1, self.data[0].shape[0])
            index_i, index_j = index_i[0:sep], index_j[sep:]

        elif self.choice == "gauss":
            sep = np.random.normal(loc=0.5, scale=0.1, size=1)
            sep = int(sep * self.data[0].shape[0])
            if x == 0:
                x = 1
            elif x >= self.data[0].shape[0]:
                x = self.data[0].shape[0]
            index_i, index_j = index_i[0:x], index_j[x:]

        min_error, max_error, expected_lp = error_cover_area(
            lp_i, lp_j, len(index_i), len(index_j), self.CI
        )

        return expected_lp, index_i, index_j, min_error, max_error

    def __getitem__(self, idx):
        data_i, label_i, lp_i = self.data[idx], self.label[idx], self.lp[idx]
        MixBag = random.choice([True, False])
        if MixBag:
            index = np.random.randint(0, self.len)
            data_j, labels_j, lp_j = (
                self.data[index],
                self.label[index],
                self.lp[index],
            )

            index_i, index_j = np.arange(data_i.shape[0]), np.arange(data_i.shape[0])
            random.shuffle(index_i)
            random.shuffle(index_j)

            expected_lp, index_i, index_j, ci_min, ci_max = self.sampling(
                index_i, index_j, lp_i, lp_j
            )

            subbag_i, subbag_labels_i = data_i[index_i], label_i[index_i]
            subbag_j, subbag_labels_j = (
                data_j[index_j],
                labels_j[index_j],
            )
            mixed_bag = np.concatenate([subbag_i, subbag_j], axis=0)
            mixed_label = np.concatenate([subbag_labels_i, subbag_labels_j])

            if len(mixed_bag.shape) == 3:
                mixed_bag = mixed_bag.reshape(
                    mixed_bag.shape[0], mixed_bag.shape[1], mixed_bag.shape[2], 1
                )

            # bs: bag size, w: width, h: height, c: channel
            (bs, w, h, c) = mixed_bag.shape
            # normalization
            empty_data = torch.zeros(bs, c, w, h)
            for i in range(bs):
                empty_data[i] = self.transform(mixed_bag[i])
            mixed_bag = empty_data

            return {
                "img": mixed_bag,  # img: [10, 3, 32, 32]
                "label": torch.tensor(mixed_label).long(),  # label: [10]
                "label_prop": torch.tensor(expected_lp).float(),  # label_prop: [10]
                "ci_min_value": torch.tensor(ci_min).float(),  # ci_min_value: [10]
                "ci_max_value": torch.tensor(ci_max).float(),  # ci_max_value: [10]
            }

        else:
            data, label, lp = self.data[idx], self.label[idx], self.lp[idx]
            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform(data[i])
            data = trans_data

            ci_min, ci_max = (
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
            )

            return {
                "img": data,  # img: [10, 3, 32, 32]
                "label": torch.tensor(label).long(),  # label: [10]
                "label_prop": torch.tensor(lp).float(),  # label_prop: [10]
                "ci_min_value": ci_min,  # ci_min_value: [10]
                "ci_max_value": ci_max,  # ci_max_value: [10]
            }


def error_cover_area(
    proportion1: list,
    proportion2: list,
    sampling_num1: int,
    sampling_num2: int,
    confidence_interval: float,
):
    a: float = sampling_num1 / (sampling_num1 + sampling_num2)
    b: float = sampling_num2 / (sampling_num1 + sampling_num2)
    t = norm.isf(q=confidence_interval)
    cover1 = t * np.sqrt(proportion1 * (1 - proportion1) / sampling_num1)
    cover2 = t * np.sqrt(proportion2 * (1 - proportion2) / sampling_num2)
    expected_plp = a * proportion1 + b * proportion2
    confidence_area = t * cover1 + b * cover2
    min = expected_plp - confidence_area
    max = expected_plp + confidence_area
    return min, max, expected_plp


def model_import(args, model_name=None):
    model = resnet18(pretrained=args.pretrained)
    if model:
        if args.channels != 3:
            model.conv1 = nn.Conv2d(
                args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.fc = nn.Linear(model.fc.in_features, args.classes)
        model = model.to(args.device)
    return model
