from argparse import ArgumentParser
from datetime import datetime
from time import time
from typing import Optional

import colossalai
import datasets
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.nn.optimizer import HybridAdam
from data_module import GLUEDataModule
from func_module import evaluate, get_mem_info, get_tflops, seed_all, train
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

from elixir.search import minimum_waste_search
from elixir.wrapper import ElixirModule, ElixirOptimizer

if __name__ == '__main__':
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()
    world_size = dist.get_world_size()
    world_group = dist.GroupMember.WORLD
    local_rank = dist.get_rank()

    parser = ArgumentParser()
    parser.add_argument('--task', default='mrpc')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2.4e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_fraction', type=float, default=0.1)
    args = parser.parse_args()

    assert args.batch_size % world_size == 0
    global_batch_size = args.batch_size
    local_batch_size = args.batch_size // world_size

    global_seed = 3407
    seed_all(global_seed)
    logger.info('Random is set to {} in all processes.'.format(global_seed), ranks=[0])

    model_name = 'bert-base-uncased'
    logger.info('Data is preparing now.', ranks=[0])
    dm = GLUEDataModule(model_name_or_path=model_name,
                        task_name=args.task,
                        train_batch_size=local_batch_size,
                        eval_batch_size=global_batch_size)
    dm.setup('fit')

    config = AutoConfig.from_pretrained(model_name, num_labels=dm.num_labels)
    metric = datasets.load_metric('glue', dm.task_name, experiment_id=datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))

    logger.info('Model is creating now.', ranks=[0])
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])
    logger.info(get_mem_info(), ranks=[0])

    logger.info('Optimizer is creating now.', ranks=[0])
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    # optimizer = HybridAdam(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    sr = minimum_waste_search(model, world_size, torch.float32, verbose=True)
    model = ElixirModule(model, sr, world_group, dtype=torch.float32, use_fused_kernels=True)
    optimizer = ElixirOptimizer(model, optimizer, init_step=True)    #, initial_scale=64)

    logger.info('Dataloder is creating now.', ranks=[0])
    train_loader, train_sampler = dm.train_loader_and_sampler()
    valid_loader = dm.val_loader_and_sampler()

    logger.info('Learning rate scheduler is creating now.', ranks=[0])
    num_epoch = args.epochs
    steps_per_epoch = len(train_loader)
    num_all_steps = num_epoch * steps_per_epoch
    num_warm_steps = int(num_all_steps * args.warmup_fraction)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=num_warm_steps,
                                                   num_training_steps=num_all_steps)

    for epoch in range(num_epoch):
        logger.info('Epoch {} starts'.format(epoch), ranks=[0])
        dist.barrier()
        train(epoch=epoch,
              sampler=train_sampler,
              model=model,
              loader=train_loader,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              show_progress=local_rank == 0,
              optimizer_backward=True)
        percentage, f1 = evaluate(model=model, metric=metric, loader=valid_loader, show_progress=local_rank == 0)
        logger.info('valid correct percentage: {:.4f}\nf1: {:.4f}'.format(percentage, f1), ranks=[0])
