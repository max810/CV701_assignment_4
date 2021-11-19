import argparse
from math import log
import os
from datetime import datetime
from time import perf_counter

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import wandb

from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.mpii import Mpii, get_mpii_validation_accuracy
from stacked_hourglass.train import do_training_epoch, do_validation_epoch
from stacked_hourglass.utils.logger import Logger
from stacked_hourglass.utils.misc import save_checkpoint, adjust_learning_rate


def main(args):
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    # Disable gradient calculations by default.
    torch.set_grad_enabled(False)

    # create checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)

    if args.arch == 'hg1':
        model = hg1(pretrained=False)
    elif args.arch == 'hg2':
        model = hg2(pretrained=False)
    elif args.arch == 'hg8':
        model = hg8(pretrained=False)
    else:
        raise Exception('unrecognised model architecture: ' + args.arch)
    
    model = DataParallel(model).to(device)
    wandb.watch(model)

    optimizer = RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)

    best_acc = 0

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # create data loader
    train_dataset = Mpii(args.image_path, is_train=True, inp_res=args.input_shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = Mpii(args.image_path, is_train=False, inp_res=args.input_shape)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # train and eval
    lr = args.lr
    cum_train_time = 0
    cum_val_time = 0
    try:
        for epoch in trange(args.start_epoch, args.epochs, desc='Overall', ascii=True):
            lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)

            # train for one epoch
            t1 = perf_counter()
            train_loss, train_acc = do_training_epoch(train_loader, model, device, Mpii.DATA_INFO,
                                                    optimizer,
                                                    acc_joints=Mpii.ACC_JOINTS)
            t2 = perf_counter()
            train_time_epoch = t2 - t1
            cum_train_time += train_time_epoch

            if (cum_train_time / 60) > 180:
                raise StopIteration

            # evaluate on validation set
            t1 = perf_counter()
            valid_loss, valid_acc, predictions = do_validation_epoch(val_loader, model, device,
                                                                    Mpii.DATA_INFO, False,
                                                                    acc_joints=Mpii.ACC_JOINTS)
            t2 = perf_counter()
            val_time_epoch = t2 - t1
            cum_val_time += val_time_epoch

            # print metrics
            tqdm.write(f'[{epoch + 1:3d}/{args.epochs:3d}] lr={lr:0.2e} '
                    f'train_loss={train_loss:0.4f} train_acc={100 * train_acc:0.2f} '
                    f'valid_loss={valid_loss:0.4f} valid_acc={100 * valid_acc:0.2f}')

            # append logger file
            logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])
            logger.plot_to_file(os.path.join(args.checkpoint, 'log.svg'), ['Train Acc', 'Val Acc'])

            # remember best acc and save checkpoint
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

            logs = {            
                'epoch': epoch + 1, 
                'epoch/val_acc': valid_acc, 
                'epoch/train_acc': train_acc,
                'epoch/loss':train_loss, 
                'epoch/val_loss':valid_loss, 
                'epoch/lr': lr,
                'epoch/train_time_m': train_time_epoch / 60.0,
                'epoch/cum_train_time_m': cum_train_time / 60.0,
                'epoch/val_time': val_time_epoch,
            }


            individual_join_accs = get_mpii_validation_accuracy(predictions)
            for k, v in individual_join_accs.items():
                logs[f'accs/{k}'] = v

            wandb.log(logs)
    
    except StopIteration:
        # Training time limit hit
        pass

    total_training_time = cum_train_time
    average_val_time = cum_val_time / args.epochs
    total_epochs = epoch

    wandb.log({
        "total_train_time_m": total_training_time / 60.0,
        "avg_val_time": average_val_time,
        "total_epochs": total_epochs,
    })

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stacked hourglass model.')
    # Dataset setting
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg8',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    # Training strategy
    parser.add_argument('--input_shape', default=(256, 256), type=int, nargs='+',
                        help='Input shape of the model. Given as: (H, W)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--debug', default=False, type=bool, metavar='BOOL',
                        help='disable wandb logging')

    args = parser.parse_args()         
    
    date = datetime.now().strftime("%b%d_%H-%M-%S")
    print(args)

    wandb.init(
        project="CV701_assignment_4",
        name=f"{date}_baseline",
        entity="max810",
        group="Baseline_1.10",
        mode='disabled' if args.debug else None
    )     

    wandb.log(vars(args))

    main(args)
