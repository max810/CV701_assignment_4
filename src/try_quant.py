import os
import copy
from datetime import datetime

import torch
import wandb
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
    mode = None
    date = datetime.now().strftime("%b%d_%H-%M-%S")
    #
    # print("Re-running experiments")
    # baseline = load_model('../checkpoint/hg2/model_best.pth.tar')
    # baseline.eval()
    #
    # wandb.init(
    #     project="CV701_assignment_4",
    #     name=f"{date}_baseline",
    #     entity="max810",
    #     group=f"Evaluation_quantization",
    #     mode=mode,
    # )
    # baseline_eval_log = run_evaluation(baseline, 'cpu', '../dataset', '')
    # num_params = sum(p.numel() for p in baseline.parameters())  # number of parameters doesn't change
    #
    # print("Size of model before quantization")
    # baseline_model_size = get_model_size_kb(baseline)
    # print(baseline_model_size)
    #
    # baseline_eval_log['params'] = num_params
    # baseline_eval_log['size_kb'] = baseline_model_size
    # wandb.log(baseline_eval_log)
    # wandb.finish()

    # Quantization
    wandb.init(
        project="CV701_assignment_4",
        name=f"{date}_quantization_old_model",
        entity="max810",
        group=f"Evaluation_quantization",
        mode=mode,
    )

    # model_to_quantize = copy.deepcopy(baseline.module)
    # qconfig = get_default_qconfig("fbgemm")
    # qconfig_dict = {"": qconfig}
    #
    # prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    # print(prepared_model.graph)
    #
    # # calibration
    # run_small_validation(prepared_model, 100)
    #
    # quantized_model = convert_fx(prepared_model)
    # print(quantized_model)
    #
    # print("Size of model after quantization")
    # quant_model_size = get_model_size_kb(quantized_model)
    # print(quant_model_size)
    #
    # torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)

    loaded_quantized_model = torch.jit.load('../fx_quantized_final.pth')

    # warmup for JIT
    print("JIT Warmup")
    run_small_validation(loaded_quantized_model, 5)

    quant_eval_log = run_evaluation(loaded_quantized_model, 'cpu', '../dataset', '')
    quant_eval_log['params'] = 6730912  # size number of params
    quant_eval_log['size_kb'] = 7428.722
    wandb.log(quant_eval_log)
