import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
from torchvision.models import resnet18
import logging
from train import train_net

from hydra.utils import to_absolute_path as to_abs_path 
from utils import *


def main(args):
    fix_seed(args.seed)

    make_folder(args.output_path)
    args.output_path += '%s/' % (args.dataset)
    make_folder(args.output_path)
    args.output_path +=   f'seed{args.seed}' + '-aug-' + str(args.augmentation) + '/'
    make_folder(args.output_path)   

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)

    train_loader, val_loader, test_loader = load_data_bags(args)

    fix_seed(args.seed)
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(args.device)

    loss_function = ProportionLoss(metric=args.loss_selection)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loss, val_loss, test_mIoU = train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
    return train_loss, val_loss, test_mIoU


if __name__ == '__main__':
    tloss, vloss, tmIoU = {}, {}, {}
    
    for bags in [80, 120, 160, 200, 240]:
        ins = 32
        for multi in [1, 2, 3, 4]:  
            for seed in range(5):
                parser = argparse.ArgumentParser()
                parser.add_argument('--seed', default=seed,
                                    type=int, help='seed value')
                parser.add_argument('--dataset', default=f'cifar10/tyofuku-3class-ins{ins}-bags{bags}-multi{multi}',
                                    type=str, help='name of dataset')
                parser.add_argument('--device', default='cuda:0',
                                    type=str, help='device')
                parser.add_argument('--split_ratio', default=0.25,
                                    type=float, help='split ratio')
                parser.add_argument('--batch_size', default=512,
                                    type=int, help='batch size for training.')
                parser.add_argument('--mini_batch', default=4,
                                    type=int, help='mini batch for training.')
                parser.add_argument('--num_epochs', default=1000, type=int,
                                    help='number of epochs for training.')
                parser.add_argument('--num_workers', default=0, type=int,
                                    help='number of workers for training.')
                parser.add_argument('--patience', default=10,
                                    type=int, help='patience of early stopping')
                parser.add_argument('--lr', default=3e-4,
                                    type=float, help='learning rate')
                parser.add_argument('--output_path',
                                    default='result/', type=str, help="output file name")
                parser.add_argument('--loss_selection',
                                    default='ce', type=str, help="loss choice")
                parser.add_argument('--num_sampled_instances',
                                    default=64, type=int, help="number of the sampled instnace")
                parser.add_argument('--classes',
                                    default=3, type=int, help="number of the sampled instnace")
                parser.add_argument('--augmentation',
                                    default=False, type=bool, help="number of the sampled instnace")
                parser.add_argument('--change_num',
                                    default=16, type=int, help="unlabel_num")
                parser.add_argument('--bags',
                                    default=bags, type=int, help="bags_num")
                parser.add_argument('--fold',
                                    default=0, type=int, help="bags_num")
                args = parser.parse_args()

                #################
                train_loss, val_loss, test_mIoU = main(args)

                tloss[f'seed{args.seed}-ins{ins}-bags{args.bags}-multi{multi}'] = train_loss
                vloss[f'seed{args.seed}-ins{ins}-bags{args.bags}-multi{multi}'] = val_loss
                tmIoU[f'seed{args.seed}-ins{ins}-bags{args.bags}-multi{multi}'] = test_mIoU

        with open(to_abs_path('result/' + args.dataset + '/add_bags_tloss.json'), 'w') as f:
            json.dump(tloss, f, indent=4)
        with open(to_abs_path('result/' + args.dataset + '/add_bags_vloss.json'), 'w') as f:
            json.dump(vloss, f, indent=4)
        with open(to_abs_path('result/' + args.dataset + '/add_bags_mIoU.json'), 'w') as f:
            json.dump(tmIoU, f, indent=4)