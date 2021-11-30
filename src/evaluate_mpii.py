import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stacked_hourglass.datasets.mpii import Mpii, print_mpii_validation_accuracy
from stacked_hourglass.train import do_validation_epoch, do_validation_step


def run_small_validation(model, num_batches, val_loader):
    i = 0

    progress = tqdm(val_loader, desc=f'Validation subset ({num_batches} batches)', total=num_batches, ascii=True,
                    leave=True)

    for input, target, meta in progress:
        if i == num_batches:
            break

        target_weight = meta['target_weight']

        do_validation_step(model, input, target, Mpii.DATA_INFO, target_weight,
                           flip=False)
        i += 1


def main(args):
    device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    model = torch.jit.load(args.model_file)
    model = model.to(device)
    # Initialise the MPII validation set dataloader.
    val_dataset = Mpii(args.image_path, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print("Warming the JIT model up...")
    run_small_validation(model, num_batches=5, val_loader=val_loader)

    # Generate predictions for the validation set.
    _, _, predictions = do_validation_epoch(val_loader, model, device, Mpii.DATA_INFO, args.flip)

    # Report PCKh for the predictions.
    print('\nFinal validation PCKh scores:\n')
    print_mpii_validation_accuracy(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--image-path', required=True, type=str,
                        help='path to MPII Human Pose images')
    # parser.add_argument('--arch', metavar='ARCH', default='hg1',
    #                     choices=['hg1', 'hg2', 'hg8'],
    #                     help='model architecture')
    parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')

    main(parser.parse_args())
