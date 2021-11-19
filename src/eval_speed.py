import os.path
from datetime import datetime
from time import perf_counter

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
import wandb

from stacked_hourglass.datasets.mpii import Mpii, get_mpii_validation_accuracy, print_mpii_validation_accuracy
from stacked_hourglass.train import do_validation_epoch


def get_model_size_kb(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e3
    os.remove('temp.p')
    return size


def run_evaluation(model, device: str, image_path: str, run_prefix: str, eval_group: str, mode: str = None,
                   extra_logs={}):
    date = datetime.now().strftime("%b%d_%H-%M-%S")

    wandb.init(
        project="CV701_assignment_4",
        name=f"{date}_{run_prefix}",
        entity="max810",
        group=f"Evaluation_{eval_group}",
        mode=mode,
    )

    model_size = get_model_size_kb(model)
    num_params = sum(p.numel() for p in model.parameters())

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    model = model.to(torch.device(device))

    # Initialise the MPII validation set dataloader.
    val_dataset = Mpii(image_path, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Generate predictions for the validation set.
    a = perf_counter()
    _, val_acc, predictions = do_validation_epoch(val_loader, model, torch.device(device), Mpii.DATA_INFO, flip=False)
    b = perf_counter()

    val_time = b - a

    logs = {}

    # Report PCKh for the predictions.
    individual_join_accs = get_mpii_validation_accuracy(predictions)
    for k, v in individual_join_accs.items():
        logs[f'accs/{k}'] = v
    logs['val_acc'] = val_acc
    logs['val_time'] = val_time

    logs['device'] = device
    logs['size_kb'] = model_size
    logs['params'] = num_params

    for k, v in extra_logs.items():
        logs[k] = v

    wandb.log(logs)

    print('\nFinal validation PCKh scores:\n')
    print_mpii_validation_accuracy(predictions)
