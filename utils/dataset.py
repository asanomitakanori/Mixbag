import random

import numpy as np
import torch
import torchvision.transforms as transforms

from utils.utils import ci_loss_interval, fix_seed, worker_init_fn


def load_data(args, stage: str):
    if stage == "train":
        train_bags = np.load(f"data/{args.dataset}/{args.fold}/train_bags.npy")
        train_labels = np.load(f"data/{args.dataset}/{args.fold}/train_labels.npy")
        train_lps = np.load(f"data/{args.dataset}/{args.fold}/train_lps.npy")
        train_dataset = Dataset_Mixbag(
            args=args,
            data=train_bags,
            label=train_labels,
            lp=train_lps,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            worker_init_fn=worker_init_fn(args.seed),
            shuffle=True,
            num_workers=args.num_workers,
        )
        return train_loader

    elif stage == "val":
        val_bags = np.load(f"data/{args.dataset}/{args.fold}/val_bags.npy")
        val_labels = np.load(f"data/{args.dataset}/{args.fold}/val_labels.npy")
        val_lps = np.load(f"data/{args.dataset}/{args.fold}/val_lps.npy")
        val_dataset = Dataset_Val(
            data=val_bags,
            label=val_labels,
            lp=val_lps,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return val_loader

    elif stage == "test":
        test_data = np.load(f"data/{args.dataset}/test_data.npy")
        test_label = np.load(f"data/{args.dataset}/test_label.npy")
        test_dataset = Dataset_Test(data=test_data, label=test_label)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return test_loader

    else:
        print("you should set stage")


class Dataset_Base(torch.utils.data.Dataset):
    """Base class for Dataset."""

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

    def transform_data(self, data):
        """Nomaralize data"""
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
        (bs, w, h, c) = data.shape
        trans_data = torch.zeros((bs, c, w, h))
        for i in range(bs):
            trans_data[i] = self.transform(data[i])
        return trans_data

    def __getitem__(self, idx):
        """Overload this function in your Dataset."""
        raise NotImplementedError


class Dataset_Mixbag(Dataset_Base):
    """
    Training Dataset. This is a MixBag dataloader,
    so we can use MixBag by applying this dataloader.
    CI loss is not applied in this module.
    """

    def __init__(self, args, data, label, lp):
        super().__init__(data, label)
        fix_seed(args.seed)
        self.lp = lp
        self.classes = args.classes
        self.choice = args.choice
        self.CI = args.confidence_interval

    def sampling(self, index: list, lp_i: float, lp_j: float):
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
                random.sample(index, len(index) // 2),
                random.sample(index, len(index) // 2),
            )

        elif self.choice == "uniform":
            sep = np.random.randint(1, self.data[0].shape[0])
            index_i = random.sample(index, sep)
            index_j = random.sample(index, len(index) - sep)

        elif self.choice == "gauss":
            sep = np.random.normal(loc=0.5, scale=0.1, size=1)
            sep = int(sep * self.data[0].shape[0])
            if sep == 0:
                sep = 1
            elif sep >= self.data[0].shape[0]:
                sep = self.data[0].shape[0]

            index_i = random.sample(index, sep)
            index_j = random.sample(index, len(index) - sep)

        ci_min, ci_max, expected_lp = ci_loss_interval(
            lp_i, lp_j, len(index_i), len(index_j), self.CI
        )

        return expected_lp, index_i, index_j, ci_min, ci_max

    def __getitem__(self, idx):
        data_i, label_i, lp_i = self.data[idx], self.label[idx], self.lp[idx]
        MixBag = random.choice([True, False])
        if MixBag:
            j = np.random.randint(0, self.len)
            data_j, labels_j, lp_j = (
                self.data[j],
                self.label[j],
                self.lp[j],
            )

            id = list(range(data_i.shape[0]))

            # expected_lp: mixed_bag's label proportion
            # id_i: index used for creating subbag_i from data_i
            # id_j: index used for creating subbag_j from data_j
            # ci_min: minimam value of confidence interval
            # ci_max: maximam value of confidence interval
            expected_lp, id_i, id_j, ci_min, ci_max = self.sampling(id, lp_i, lp_j)

            subbag_i, subbag_labels_i = data_i[id_i], label_i[id_i]
            subbag_j, subbag_labels_j = (data_j[id_j], labels_j[id_j])

            mixed_bag = np.concatenate([subbag_i, subbag_j], axis=0)
            mixed_label = np.concatenate([subbag_labels_i, subbag_labels_j])

            # bs: bag size, w: width, h: height, c: channel
            mixed_bag = self.transform_data(mixed_bag)

            return {
                "img": mixed_bag,  # img: [10, 3, 32, 32]
                "label": torch.tensor(mixed_label).long(),  # label: [10]
                "label_prop": torch.tensor(expected_lp).float(),  # label_prop: [10]
                "ci_min_value": torch.tensor(ci_min).float(),  # ci_min_value: [10]
                "ci_max_value": torch.tensor(ci_max).float(),  # ci_max_value: [10]
            }

        else:
            data, label, lp = self.data[idx], self.label[idx], self.lp[idx]
            data = self.transform_data(data)

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


class Dataset_Val(Dataset_Base):
    """Validation Dataset"""

    def __init__(self, data, label, lp):
        super().__init__(data, label)
        self.lp = lp

    def __getitem__(self, idx):
        data, label, lp = self.data[idx], self.label[idx], self.lp[idx]
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
        # bs: bag size, w: width, h: height, c: channel
        data = self.transform_data(data)

        return {
            "img": data,
            "label": torch.tensor(label).long(),
            "label_prop": torch.tensor(lp).float(),
        }


class Dataset_Test(Dataset_Base):
    """Test Dataset"""

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
