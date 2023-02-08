import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from torchvision.models import resnet18
import logging
from train import train_net

from hydra.utils import to_absolute_path as to_abs_path 
from utils2 import *


def main(args):
    fix_seed(args.seed)

    make_folder(args.output_path)
    args.output_path += '%s/' % (args.dataset)
    make_folder(args.output_path)
    args.output_path +=   f'ins_aug' + f'/ins_augtype_{args.augmentation}-ins{args.ins}-bags{args.bags}/'
    make_folder(args.output_path)   

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)

    train_loader, val_loader, test_loader = load_data_bags(args)

    fix_seed(args.seed)
    model = resnet18(pretrained=True)
    if args.channels != 3:
        model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, args.classes)
    model = model.to(args.device)

    loss_function = ProportionLoss(metric=args.loss_selection)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loss, val_loss, test_acc = train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
    return train_loss, val_loss, test_acc


if __name__ == '__main__':
    tloss, vloss, tmIoU = {}, {}, {}
    test_list = []
    for fold in range(5):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--ins', default=32,
                            type=int, help='number of bags_size')
        parser.add_argument('--bags', default=1024,
                            type=int, help='number of bags_num')
        parser.add_argument('--dataset', default=f'cifar10',
                            type=str, help='name of dataset')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--split_ratio', default=0.25,
                            type=float, help='split ratio')
        parser.add_argument('--batch_size', default=512,
                            type=int, help='batch size for training.')
        parser.add_argument('--mini_batch', default=32,
                            type=int, help='mini batch for training.')
        parser.add_argument('--num_epochs', default=1000, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--num_workers', default=4, type=int,
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
                            default=10, type=int, help="number of the sampled instnace")
        parser.add_argument('--channels',
                            default=3, type=int, help="input_channels")
        parser.add_argument('--augmentation',
                            default='flip', type=str, help="augmentation")
        parser.add_argument('--change_num',
                            default=16, type=int, help="unlabel_num")
        parser.add_argument('--fold',
                            default=fold, type=int, help="bags_num")
        args = parser.parse_args()
        args.dataset = args.dataset + f'/{args.classes}class-ins{args.ins}-bags{args.bags}'
        #################
        train_loss, val_loss, test_acc = main(args)
        test_list.append(test_acc)

    if os.path.exists(to_abs_path('result/' + args.dataset + f'/ins_aug_tloss.json')):
        with open(to_abs_path('result/' + args.dataset + f'/ins_aug_tloss.json'), 'r') as f:
            tloss = json.load(f)
        with open(to_abs_path('result/' + args.dataset + f'/ins_aug_vloss.json'), 'r') as f:
            vloss = json.load(f)
        with open(to_abs_path('result/' + args.dataset + f'/ins_aug_mIoU.json'), 'r') as f:
            tmIoU = json.load(f)

    tloss[f'ins_augtype_{args.augmentation}-ins{args.ins}-bags{args.bags}'] = train_loss
    vloss[f'ins_augtype_{args.augmentation}-ins{args.ins}-bags{args.bags}'] = val_loss
    tmIoU[f'ins_augtype_{args.augmentation}-ins{args.ins}-bags{args.bags}'] = np.mean(test_list)

    with open(to_abs_path('result/' + args.dataset + f'/ins_aug_tloss.json'), 'w') as f:
        json.dump(tloss, f, indent=4)
    with open(to_abs_path('result/' + args.dataset + f'/ins_aug_vloss.json'), 'w') as f:
        json.dump(vloss, f, indent=4)
    with open(to_abs_path('result/' + args.dataset + f'/ins_aug_mIoU.json'), 'w') as f:
        json.dump(tmIoU, f, indent=4)
    
