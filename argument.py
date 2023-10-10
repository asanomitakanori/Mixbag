from tap import Tap


class SimpleArgumentParser(Tap):
    # Training
    seed: int = 0  # seed value
    epochs: int = 1000  # the number of epochs
    batch_size: int = 512  # batch size for training
    mini_batch: int = 32  # mini batch size for training (the number of labeled bags)
    patience: int = 1  # patience of early stopping
    lr: float = 3e-4  # learning rate
    confidence_interval: float = 0.005  # 0.005 means 99% confidential interval
    choice: str = "uniform"  # γ-sampling

    # Dataset
    bag_size: int = 10  # bag size
    bags_num: int = 512  # the number of labeled bags
    num_workers: int = 4  # number of workers for dataloader

    # Model
    pretrained: bool = True

    classes: int = 10
    channels: int = 3  # input image's channel
    dataset: str = "cifar10"  # dataset name
    output_path: str = "result/"  # output file name
    device: str = "cuda:0"  # device
