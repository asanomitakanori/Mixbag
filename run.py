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
from utils import *
from losses import *

from val import eval_net
from test import test_net


def train_net(args,
              model
              ):

    fix_seed(args.seed)

    train_loader, val_loader, test_loader = load_data_bags(args)

    criterion_train = ProportionLoss_CI()
    criterion_val = ProportionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logging.info(f'''Starting training:
        Epochs:                {args.epochs}
        Mini Batch size:       {args.mini_batch}
        Learning rate:         {args.lr}
        Dataset:               {args.dataset}
        Bag size:              {args.bag_size}
        Bag Num:               {args.bags_num}
        Training size:         {len(train_loader)}
        Validation size:       {len(val_loader)}
        Test size:             {len(test_loader)}
        Checkpoints:           {args.output_path + args.fold}
        Device:                {args.device}
        Optimizer              {optimizer.__class__.__name__}
        Confidence Interval:   {args.confidence_interval}
    ''')

    train_loss, val_loss = [], []
    train_acc, val_acc, test_acc = [], [], []
    best_val_loss = float('inf')
    cnt = 0
    ############ Trainning ############
    for epoch in range(args.epochs):

        model.train()
        losses = []
        gt, pred = [], []
        for iteration, (data, label, lp, min_point, max_point) in enumerate(tqdm(train_loader, leave=False)):
            (b, n, c, w, h) = data.size()
            data = data.reshape(-1, c, w, h)
            label = label.reshape(-1)
            data, lp = data.to(args.device), lp.to(args.device)

            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(b, n, -1)
            pred_prop = confidence.mean(dim=1)

            loss = criterion_train(pred_prop, 
                                   lp, 
                                   min_point.to(args.device), 
                                   max_point.to(args.device)
                                   )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        train_loss = np.array(losses).mean()
        gt, pred = np.array(gt), np.array(pred)
        train_acc = (gt == pred).mean()

        logging.info(f'[Epoch: {epoch+1}/{args.epochs}] train loss: {np.round(train_loss, 4)}, acc: {np.round(train_acc, 4)}')

        ############ Validation ############
        val_loss, val_acc = eval_net(args, 
                                     epoch, 
                                     model, 
                                     val_loader,
                                     criterion_val)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            cnt = 0
            best_epoch = epoch
            best_path = args.output_path +  args.fold + f'/CP_epoch{best_epoch + 1}.pkl'
            torch.save(model.state_dict(), 
                       best_path)
        else:
            cnt += 1
            if args.patience == cnt:
                break
    
    ############ Test ############
    model.load_state_dict(
        torch.load(best_path, map_location=args.device)
    )
    logging.info(f'Model loaded from {best_path}')
    
    test_acc, test_cm = test_net(args, 
                                 epoch, 
                                 model, 
                                 test_loader)

    logging.info('[Epoch: %d/%d] test acc: %.4f' %(epoch+1, args.epochs, test_acc))
    logging.info('===============================')
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
            train_net(args, 
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