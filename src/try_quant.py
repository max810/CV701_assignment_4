import os
import copy

import torch
from torch.ao.quantization.qconfig import get_default_qconfig
from torch.nn.parallel.data_parallel import DataParallel
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_speed import run_evaluation, get_model_size_kb
from stacked_hourglass import hg2
from stacked_hourglass.datasets.mpii import Mpii
from stacked_hourglass.train import do_validation_step

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

torch.set_grad_enabled(False)

rerun = True

val_dataset = Mpii('../dataset', is_train=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False,
                        num_workers=4, pin_memory=True)

fx_graph_mode_model_file_path = "../fx_quantized_final.pth"

extra_logs = {}


def run_small_validation(model, num_batches):
    i = 0

    progress = tqdm(val_loader, desc=f'Validation subset ({num_batches} batches)', total=num_batches, ascii=True, leave=True)

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
    if rerun:
        print("Re-running experiments")
        baseline = load_model('../checkpoint/hg2/model_best.pth.tar')
        baseline.eval()
        model_to_quantize = copy.deepcopy(baseline.module)
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
        print(prepared_model.graph)

        # calibration
        run_small_validation(prepared_model, 100)

        # run_evaluation(prepared_model, 'cpu', '../dataset', '---', '---', 'disabled')

        quantized_model = convert_fx(prepared_model)
        print(quantized_model)

        print("Size of model before quantization")
        print(get_model_size_kb(baseline))
        print("Size of model after quantization")
        model_size = get_model_size_kb(quantized_model)
        print(model_size)

        extra_logs['size_kb'] = model_size
        extra_logs['params'] = sum(p.numel() for p in baseline.parameters())  # number of parameters doesn't change

        torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)

    loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

    # warmup for JIT
    print("JIT Warmup")
    run_small_validation(loaded_quantized_model, 5)

    run_evaluation(loaded_quantized_model, 'cpu', '../dataset', 'quant_simple', 'Quantization', extra_logs=extra_logs)
