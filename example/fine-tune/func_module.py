import random

import numpy as np
import psutil
import torch
import torch.distributed as dist
from colossalai.utils import get_current_device
from tqdm import tqdm


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB'.format(
        prefix, get_cur_gpu_mem(), get_gpu_mem(), get_cpu_mem())


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def train(epoch, sampler, model, loader, optimizer, show_progress=True, lr_scheduler=None, optimizer_backward=False):
    if sampler:
        sampler.set_epoch(epoch)
    model.train()
    train_iter = iter(loader)
    num_steps_per_epoch = len(loader)

    def run_step():
        batch = next(train_iter)
        for key, val in batch.items():
            batch[key] = val.cuda()

        optimizer.zero_grad()
        outputs = model(**batch)
        output_loss = outputs[0]
        step_loss = output_loss.item()
        if optimizer_backward:
            optimizer.backward(output_loss)
        else:
            output_loss.backward()
        optimizer.step()
        return step_loss

    with tqdm(range(num_steps_per_epoch), desc='train', ncols=0, disable=not show_progress) as t:
        for step in t:
            loss = run_step()
            lr_scheduler.step()
            t.set_postfix(loss=f'{loss:.4f}')

    try:
        while True:
            next(train_iter)
    except StopIteration:
        pass


def evaluate(model, loader, metric, show_progress=True):
    model.eval()
    valid_iter = iter(loader)
    num_steps_per_epoch = len(loader)

    with torch.no_grad():
        with tqdm(range(num_steps_per_epoch), desc='valid', ncols=0, disable=not show_progress) as t:
            for step in t:
                batch = next(valid_iter)
                for key, val in batch.items():
                    batch[key] = val.cuda()

                outputs = model(**batch)
                val_loss, logits = outputs[:2]
                preds = torch.argmax(logits, dim=-1)
                labels = batch['labels']
                metric.add_batch(predictions=preds, references=labels)

    try:
        while True:
            next(valid_iter)
    except StopIteration:
        pass

    score = metric.compute()
    return score['accuracy'], score['f1']
