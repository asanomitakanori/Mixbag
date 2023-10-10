import os
import sys
import logging
from tqdm import tqdm
from argument import SimpleArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from hydra.utils import to_absolute_path as abs_path
from utils.utils import *
from utils.losses import *

from dust.test import test_net

from train import Run


def net(args):
    fix_seed(args.seed)
    run = Run(args)

    # Training & Validation
    for epoch in range(args.epochs):
        run.train(args, epoch)
        run.val(args, epoch)

        break_flag = run.early_stopping(args, epoch)
        if break_flag:
            break
    # Test
    run.test(args, epoch)


def main(args):
    fix_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {args.device}")

    args.output_path = args.output_path + args.dataset

    # 5 fold cross validation
    for fold in range(5):
        args.fold = fold
        os.makedirs(args.output_path + str(args.fold)) if os.path.exists(
            args.output_path + str(args.fold)
        ) is False else None

        try:
            net(args)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    main(args)
