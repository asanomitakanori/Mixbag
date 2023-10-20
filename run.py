import os
import sys
import logging
import arguments

import torch

from utils.utils import *
from utils.losses import *

from trainer import Run


def main(args):
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


if __name__ == "__main__":
    args = arguments.ARGS
    fix_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_path = args.output_path + args.dataset
    print(f"Using device {args.device}")

    # 5 fold cross validation
    for fold in range(5):
        args.fold = fold
        path = args.output_path + "/" + str(args.fold)
        os.makedirs(path) if os.path.exists(path) is False else None

        try:
            main(args)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    main(args)
