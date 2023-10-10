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


def net(args, model):
    fix_seed(args.seed)
    run = Run(args)

    for epoch in range(args.epochs):
        # Training & Validation
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

    model = resnet18(pretrained=args.pretrained)
    if args.channels != 3:
        model.conv1 = nn.Conv2d(
            args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, args.classes)
    model = model.to(args.device)

    logging.info(
        f"Network:\n"
        f"\t{args.channels} input channels\n"
        f"\t{args.classes} output channels\n"
    )
    args.output_path = args.output_path + args.dataset

    for fold in range(5):
        args.fold = fold
        os.makedirs(args.output_path + str(args.fold)) if os.path.exists(
            args.output_path + str(args.fold)
        ) is False else None

        try:
            net(args, model)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), abs_path("INTERRUPTED.pth"))
            logging.info("Saved interrupt")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    main(args)
