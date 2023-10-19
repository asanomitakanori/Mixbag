# from tap import Tap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--epochs", default=1000, type=int, help="max epoch")
parser.add_argument(
    "--batch_size_test",
    default=512,
    type=int,
    help="batch size in test time. this is not applied in training",
)
parser.add_argument(
    "--mini_batch",
    default=32,
    type=int,
    help="you can set the number of labeled bags used in an iteration",
)
parser.add_argument(
    "--patience", default=10, type=int, help="patience of early stopping"
)
parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
parser.add_argument(
    "--confidence_interval",
    default=0.005,
    type=float,
    help="0.005 means 99% confidential interval",
)
parser.add_argument("--choice", default="uniform", type=str, help="γ-sampling")

parser.add_argument(
    "--num_workers", default=4, type=int, help="number of workers for dataloader"
)
# model parameter
parser.add_argument("--pretrained", default=True, type=bool)
parser.add_argument("--classes", default=10, type=int, help="the number of classes")
parser.add_argument("--channels", default=3, type=int, help="input image's channel")
parser.add_argument(
    "--dataset",
    default="cifar10",
    choices=[
        "bloodmnist",
        "octmnist",
        "organamnist",
        "orancmnist",
        "organsmnist",
        "pathmnist",
    ],
    type=str,
    help="dataset name",
)
parser.add_argument(
    "--output_path", default="result/", type=str, help="output file name"
)
parser.add_argument("--device", default="cuda:0", type=str)

ARGS = parser.parse_args()
