import os

import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval_speed import run_evaluation
from stacked_hourglass import hg2
from stacked_hourglass.datasets.mpii import Mpii
from stacked_hourglass.train import do_validation_step

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

torch.set_grad_enabled(False)

rerun = True

val_dataset = Mpii('dataset', is_train=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False,
                        num_workers=4, pin_memory=True)

fx_graph_mode_model_file_path = "../fx_quantized_final_1.pth"


def run_small_validation(model, num_batches):
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


def load_model(path):
    print('Loading model weights from file: {}'.format(path))
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model = hg2(pretrained=False)
    if sorted(state_dict.keys())[0].startswith('module.'):
        model = DataParallel(model)
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    loaded_quantized_model = torch.jit.load('fx_quantized_final.pth')

    # warmup for JIT
    print("JIT Warmup")
    run_small_validation(loaded_quantized_model, 5)

    run_evaluation(loaded_quantized_model, 'cpu', 'dataset', '')
