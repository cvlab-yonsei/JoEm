import argparse
import collections
import torch
import numpy as np
import torch.nn as nn

import model.model as module_arch
import model.metric as module_metric
import model.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from data_loader import get_seen_idx, get_unseen_idx, VOC

from utils.parse_config import ConfigParser
from trainer.trainer_pascal_zs3setting import Trainer
from model.sync_batchnorm import SynchronizedBatchNorm2d

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = config.get_logger('train')

    logger.info(f'Seed: {SEED} / {torch.randint(0, 100, (1, 1))}')

    # setup data_loader instances
    n_class = config['data_loader']['args']['n_unseen_classes']
    unseen_classes_idx = get_unseen_idx(n_class)
    ignore_bg = config['data_loader']['args']['train']['args']['ignore_bg']
    seen_classes_idx = get_seen_idx(n_class, ignore_bg=ignore_bg)
    info = {"unseen_classes_idx": unseen_classes_idx,
            "seen_classes_idx": seen_classes_idx}
    dataset = config.init_obj('data_loader', module_data, **info)

    setting_info = '\n' + '-' * 30 + '\n'
    setting_info += 'Number of images in train: %d\n' % (len(dataset.train_set))
    setting_info += '\t Remove training samples : %s\n' % (dataset.train_set.remv_unseen_img)
    setting_info += '\t Ignore background class : %s\n' % (dataset.train_set.ignore_bg)
    setting_info += '\t Ignore unseen classes   : %s\n' % (dataset.train_set.ignore_unseen)
    logger.info(setting_info)

    setting_info = '\n\nNumber of images in train: %d\n' % (len(dataset.val_set))
    setting_info += '\t Remove training samples : %s\n' % (dataset.val_set.remv_unseen_img)
    setting_info += '\t Ignore background class : %s\n' % (dataset.val_set.ignore_bg)
    setting_info += '\t Ignore unseen classes   : %s\n' % (dataset.val_set.ignore_unseen)
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

    # Print Learnable Parameters
    layer_info = '\n' + '-' * 30 + '\nUnfreezed layer_name\n'
    layer_info += '\n%s\n' % ('Feature Extractor')
    for name, param in visual_encoder.named_parameters():
        if param.requires_grad:
            layer_info += '\t%s\n' % (name)
    layer_info += '\n%s\n' % ('semantic_encoder')
    for name, param in semantic_encoder.named_parameters():
        if param.requires_grad:
            layer_info += '\t%s\n' % (name)
    logger.info(layer_info)

    # Print BatchNorm is Training or Not
    bn_info = '\n' + '-' * 30 + '\nBatchNorm Freezed\nlayer_name | training\n'
    for name, m in visual_encoder.named_modules():
        if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
            bn_info += '\t%s | %s\n' % (name, m.training)
    bn_info += '-' * 30 + '\n'
    logger.info(bn_info)

    unseen_log = '\n' + '-' * 30 + '\nUnseen Classes\n'
    for idx in unseen_classes_idx:
        unseen_log += '\t%02d - %s\n' % (int(idx), VOC[int(idx)])
    logger.info(unseen_log)

    seen_log = '\nSeen Classes\n'
    for idx in seen_classes_idx:
        seen_log += '\t%02d - %s\n' % (int(idx), VOC[int(idx)])
    seen_log += '-' * 30 + '\n'
    logger.info(seen_log)

    info = [dataset.num_classes, seen_classes_idx, unseen_classes_idx]
    evaluator = config.init_obj('evaluator', module_metric, *info)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = [
        {"params": visual_encoder.get_1x_lr_params()},
        {"params": visual_encoder.get_10x_lr_params(), "lr": config["optimizer"]["args"]["lr"] * 10},
    ]
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer_gen = config.init_obj('optimizer_gen', torch.optim, semantic_encoder.parameters())

    info = {"base_lr": config['optimizer']['args']['lr'],
            "num_epochs": config["trainer"]['epochs'],
            "iters_per_epoch": len(train_loader)}
    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, **info)

    trainer = Trainer(
        visual_encoder, semantic_encoder,
        optimizer, optimizer_gen,
        evaluator,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler=lr_scheduler,
        embeddings=dataset.embeddings
    )
    
    trainer.train()
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-re', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n', '--name'], type=str, target='name'),
        CustomArgs(['--sd', '--seed'], type=int, target='seed'),
        
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--lr_gen', '--learning_rate_gen'], type=float, target='optimizer_gen;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--nw', '--num_workers'], type=int, target='data_loader;args;num_workers'),

        CustomArgs(['--n_unseen', '--n_unseen_classes'], type=int, target='data_loader;args;n_unseen_classes'),
        CustomArgs(['--r', '--ratio'], type=float, target='hyperparameter;ratio'),
        CustomArgs(['--l', '--lamb'], type=float, target='hyperparameter;lamb'),
        CustomArgs(['--s', '--sigma'], type=float, target='hyperparameter;sigma'),
        CustomArgs(['--t', '--temp'], type=float, target='hyperparameter;temperature')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
