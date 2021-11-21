import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import wandb
from torch.nn.utils import prune
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

from eval_speed import run_evaluation, get_model_size_kb
from stacked_hourglass import hg2
from stacked_hourglass.datasets.mpii import Mpii

torch.set_grad_enabled(False)

val_dataset = Mpii('../dataset', is_train=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False,
                        num_workers=4, pin_memory=True)

fx_graph_mode_model_file_path = "../fx_quantized_final.pth"

extra_logs = {}


def get_model_size_compressed(model):
    torch.save(model.state_dict(), "/tmp/temp.p")
    os.system('gzip -qf /tmp/temp.p')
    size = os.path.getsize("/tmp/temp.p.gz") / 1e3
    os.remove('/tmp/temp.p.gz')

    return size


def load_model(path):
    print('Loading model weights from file: {}'.format(path))
    checkpoint = torch.load(path, map_location='cuda')
    state_dict = checkpoint['state_dict']
    model = hg2(pretrained=False)
    if sorted(state_dict.keys())[0].startswith('module.'):
        model = DataParallel(model)
    model.load_state_dict(state_dict, strict=True)

    return model


def run_baseline(baseline, mode):
    baseline.eval()
    date = datetime.now().strftime("%b%d_%H-%M-%S")

    wandb.init(
        project="CV701_assignment_4",
        name=f"{date}_baseline",
        entity="max810",
        group=f"Evaluation_Pruning",
        mode=mode,
    )
    logs = {}
    print(f"Baseline")
    logs['proportion'] = 0.0
    logs['size_uncompressed']: get_model_size_kb(baseline)

    eval_logs = run_evaluation(baseline, 'cuda', '../dataset', 'proportion')

    eval_logs['size_compressed'] = get_model_size_compressed(baseline)
    wandb.log(eval_logs)

    wandb.finish()


def l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model


def prune_model_l1_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    return model


def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model


if __name__ == '__main__':
    mode = None
    baseline = load_model('../checkpoint/hg2/model_best.pth.tar')
    baseline.eval()
    run_baseline(baseline, mode)

    funcs = [l1_unstructured, prune_model_global_unstructured, prune_model_l1_structured]
    for prune_func in funcs:
        run_name = prune_func.__name__
        print(run_name)
        date = datetime.now().strftime("%b%d_%H-%M-%S")

        wandb.init(
            project="CV701_assignment_4",
            name=f"{date}_{run_name}",
            entity="max810",
            group=f"Evaluation_Pruning",
            mode=mode,
        )

        logs = {}
        for proportion in np.linspace(0.1, 0.9, 17):  # 0.1, 0.15, ..., 0.85, 0.9
            print(f"Pruning {run_name}, propotion {proportion}")
            logs['proportion/proportion'] = proportion

            pruned_model = prune_func(baseline, nn.Conv2d, proportion)
            eval_logs = run_evaluation(pruned_model, 'cuda', '../dataset', 'proportion')

            logs.update(eval_logs)
            logs['size_uncompressed'] = get_model_size_kb(pruned_model)
            logs['size_compressed'] = get_model_size_compressed(pruned_model)

            wandb.log(logs)
