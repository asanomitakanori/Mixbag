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

from train import train_net
from val import eval_net
from test import test_net


def net(args,
        model
        ):

    fix_seed(args.seed)
   
    # Generating dataloader
    train_loader, val_loader, test_loader = load_data_bags(args)

    criterion_train = ProportionLoss_CI()  # Proportion loss with confidence interval
    criterion_val = ProportionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logging.info(f'''Start training:
        Epochs:                {args.epochs}
        Patience:              {args.patience}
        Mini Batch size:       {args.mini_batch}
        Learning rate:         {args.lr}
        Dataset:               {args.dataset}
        Bag size:              {args.bag_size}
        Bag Num:               {args.bags_num}
        Training size:         {len(train_loader)}
        Validation size:       {len(val_loader)}
        Test size:             {len(test_loader)}
        Checkpoints:           {args.output_path + str(args.fold)}
        Device:                {args.device}
        Optimizer              {optimizer.__class__.__name__}
        Confidence Interval:   {args.confidence_interval}
    ''')

    best_val_loss = float('inf')
    cnt = 0
    for epoch in range(args.epochs):
        # Trainning
        train_loss, train_acc = train_net(args,
                                          model,
                                          train_loader,
                                          epoch,
                                          optimizer,
                                          criterion_train
                                          )

        # Validation 
        val_loss, val_acc = eval_net(args, 
                                     epoch, 
                                     model, 
                                     val_loader,
                                     criterion_val)

        # Early Stopping 
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            cnt = 0
            best_epoch = epoch
            best_path = args.output_path +  str(args.fold) + f'/CP_epoch{best_epoch + 1}.pkl'
            torch.save(model.state_dict(), 
                       best_path)
        else:
            cnt += 1
            if args.patience == cnt:
                break
    
    # Load Best Parameters 
    model.load_state_dict(
        torch.load(best_path, map_location=args.device)
    )
    logging.info(f'Model loaded from {best_path}')

    # Test 
    test_acc, test_cm = test_net(args, 
                                 epoch, 
                                 model, 
                                 test_loader)

    save_confusion_matrix(cm=test_cm, 
                          path=args.output_path  + '/Confusion_matrix.png',
                          title='test: acc: %.4f' % test_acc)
    


def main(args):
    fix_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {args.device}')

    model = resnet18(pretrained=args.pretrained)
    if args.channels != 3:
        model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, args.classes)
    model = model.to(args.device)

    logging.info(f'Network:\n'
                 f'\t{args.channels} input channels\n'
                 f'\t{args.classes} output channels\n'
                 )
    args.output_path = args.output_path + args.dataset  
    
    for fold in range(5):
        args.fold = fold
        os.makedirs(args.output_path + str(args.fold)) if os.path.exists(args.output_path + str(args.fold)) is False else None

        try:
            net(args, 
                model)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), abs_path('INTERRUPTED.pth'))
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
                

if __name__ == '__main__':
    args = SimpleArgumentParser().parse_args()
    main(args)
