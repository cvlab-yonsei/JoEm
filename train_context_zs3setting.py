import argparse
import collections
import torch
import random
import numpy as np
import torch.nn as nn

import model.model as module_arch
import model.metric as module_metric
import model.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from data_loader import get_seen_idx, get_unseen_idx, CONTEXT59

from utils.parse_config import ConfigParser
from trainer.trainer_context_zs3setting import Trainer
from model.sync_batchnorm import SynchronizedBatchNorm2d

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    logger.info(f'Seed: {SEED} / {torch.randint(0, 100, (1, 1))}')

    # setup data_loader instances
    n_class = config['data_loader']['args']['n_unseen_classes']
    ignore_bg = config['data_loader']['args']['train']['args']['ignore_bg']
    dataset = config.init_obj(
        'data_loader',
        module_data,
        **{"unseen_classes_idx": get_unseen_idx(n_class, dataset='context59'),
           "seen_classes_idx": get_seen_idx(n_class, ignore_bg=ignore_bg, dataset='context59')}
    )

    setting_info = '\n' + '-' * 30 + '\n'
    setting_info += 'Number of images in train: %d\n' % (len(dataset.train_set))
    setting_info += '\t Remove training samples : %s\n' % (dataset.train_set.remv_unseen_img)
    setting_info += '\t Ignore background class : %s\n' % (dataset.train_set.ignore_bg)
    setting_info += '\t Ignore unseen classes   : %s\n' % (dataset.train_set.ignore_unseen)
    setting_info += '-' * 30 + '\n'
    logger.info(setting_info)

    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()

    # build model architecture, then print to console
    if config["n_gpu"] > 1:
        info = {"sync_bn": True}
    else:
        info = {"sync_bn": False}

    visual_encoder = config.init_obj('arch', module_arch, **info)
    logger.info(visual_encoder)

    semantic_encoder = config.init_obj('arch_gen', module_arch)
    logger.info(semantic_encoder)

    unseen_log = '\n' + '-' * 30 + '\nUnseen Classes\n'
    for idx in get_unseen_idx(n_class, dataset='context59'):
        unseen_log += '\t%02d - %s\n' % (int(idx), CONTEXT59[int(idx)])
    logger.info(unseen_log)

    seen_log = '\nSeen Classes\n'
    for idx in get_seen_idx(n_class, ignore_bg=ignore_bg, dataset='context59'):
        seen_log += '\t%02d - %s\n' % (int(idx), CONTEXT59[int(idx)])
    seen_log += '-' * 30 + '\n'
    logger.info(seen_log)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj(
        'optimizer',
        torch.optim,
        [{"params": visual_encoder.get_1x_lr_params()},
         {"params": visual_encoder.get_10x_lr_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
    )
    optimizer_gen = config.init_obj(
        'optimizer_gen',
        torch.optim,
        semantic_encoder.parameters()
    )

    lr_scheduler = config.init_obj(
        'lr_scheduler',
        module_lr_scheduler,
        **{"base_lr": config['optimizer']['args']['lr'],
           "num_epochs": config["trainer"]['epochs'],
           "iters_per_epoch": len(train_loader)}
    )

    evaluator = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.num_classes, get_seen_idx(n_class, ignore_bg=ignore_bg, dataset='context59'), get_unseen_idx(n_class, dataset='context59')]
    )

    trainer = Trainer(
        visual_encoder, semantic_encoder,
        optimizer, optimizer_gen,
        evaluator,
        config=config,
        train_loader=train_loader,
        val_loader=None,
        test_loader=val_loader,
        lr_scheduler=lr_scheduler,
        embeddings=dataset.embeddings
    )

    if config['test'] is not True:
        trainer.train()

    trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='JoEm')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--test', action='store_true', help='')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--n_unseen'], type=int, target='data_loader;args;n_unseen_classes'),
        CustomArgs(['--ratio'], type=float, target='hyperparameter;ratio'),
        CustomArgs(['--temp'], type=float, target='hyperparameter;temperature'),
        CustomArgs(['--alpha'], type=float, target='hyperparameter;alpha'),
        CustomArgs(['--sigma'], type=float, target='hyperparameter;sigma'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
