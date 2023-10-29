# from tap import Tap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--s", "--seed", dest="seed", default=0, type=int)
parser.add_argument(
    "--e", "--epochs", dest="epochs", default=1000, type=int, help="max epoch"
)
parser.add_argument(
    "--t_fold",
    "--t_fold",
    dest="t_fold",
    default=5,
    type=int,
    help="N-fold cross validation",
)
parser.add_argument(
    "--b",
    "--batch_size_test",
    dest="batch_size_test",
    default=512,
    type=int,
    help="batch size in test time. this is not applied in training",
)
parser.add_argument(
    "--con",
    "--consistency",
    dest="consistency",
    default="none",
    type=str,
    choices=["none", "vat", "pi"],
    help="batch size in test time. this is not applied in training",
)
parser.add_argument(
    "--m",
    "--batch_size",
    dest="batch_size",
    default=32,
    type=int,
    help="you can set the number of labeled bags used in an iteration",
)
parser.add_argument(
    "--p",
    "--patience",
    dest="patience",
    default=10,
    type=int,
    help="patience of early stopping",
)
parser.add_argument(
    "--lr",
    "--learning_rate",
    dest="lr",
    default=3e-4,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "--ci",
    "--confidence_interval",
    dest="confidence_interval",
    default=0.005,
    type=float,
    help="0.005 means 99% confidential interval",
)
parser.add_argument(
    "--choice", dest="choice", default="uniform", type=str, help="γ-sampling"
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="number of workers for dataloader"
)
# model parameter
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    default=True,
    type=bool,
    help="whether you use preterained model (True of False)",
)
parser.add_argument(
    "--classes",
    dest="classes",
    default=10,
    type=int,
    help="the number of dataset's classes. This depends on dataset.",
)
parser.add_argument(
    "--channels",
    dest="channels",
    default=3,
    type=int,
    help="Input image's channels. This depends on dataset.",
)
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="pathmnist",
    choices=[
        "cifar10",
        "svhn",
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
    "--output_path",
    dest="output_path",
    default="result/",
    type=str,
    help="output path",
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda:0",
    type=str,
    help="You can choose gpu or cpu. default is set as cuda.",
)

ARGS = parser.parse_args()
