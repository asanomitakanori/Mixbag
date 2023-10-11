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
    """
    Args:
        cm (marix):confusion matrix
        path (str): output_path
        title (str): the title of output image

    Returns:
        None
    """
    sns.heatmap(cm, annot=True, cmap="Blues_r", fmt=".4f")
    plt.xlabel("pred")
    plt.ylabel("GT")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def ci_loss_interval(
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
