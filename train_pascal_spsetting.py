import argparse
import collections
import torch
import numpy as np
import torch.nn as nn

import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
import model.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from data_loader import get_seen_idx, get_unseen_idx, VOC

from parse_config import ConfigParser
from model.sync_batchnorm import SynchronizedBatchNorm2d
from trainer.trainer_pascal_spsetting import Trainer

import json
import requests
from pandas.io.json import json_normalize

URL         = 'https://slack.com/api/chat.postMessage'
SLACK_TOKEN = 'xoxb-116215949047-1731607340946-BNe4GC18rHG6I6ioY10r00vy'
CHANNEL_ID  = 'UKDHE8QLV'

# fix random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, RATIO, LAMBDA, THRES):
    logger = config.get_logger('train')
    logger.info('**Train transform scale - (0.5, 1.5)**')
    logger.info('**Batchsize = 24**')
    logger.info('**W33 + F33 / norm with each max**')

    # setup data_loader instances
    n_class = config['data_loader']['args']['n_unseen_classes']
    unseen_classes_idx = get_unseen_idx(n_class)
    ignore_bg = config['data_loader']['args']['train']['args']['ignore_bg']
    seen_classes_idx = get_seen_idx(n_class, ignore_bg=ignore_bg)
    info = {"unseen_classes_idx": unseen_classes_idx,
            "seen_classes_idx": seen_classes_idx}
    dataset = config.init_obj('data_loader', module_data, **info)

    if dataset.train_set.ignore_bg:
        setting = 'SPNet'
    else:
        setting = 'ZS3Net'
    logger.info('Number of images in train(%s): %d' % (setting, len(dataset.train_set)))

    if dataset.val_set.ignore_bg:
        setting = 'SPNet'
    else:
        setting = 'ZS3Net'
    logger.info('Number of images in val(%s): %d' % (setting, len(dataset.val_set)))

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

    # Print BatchNorm is Training or Not
    layer_info = '\nlayer_name | training\n'
    for name, m in visual_encoder.named_modules():
        if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
            layer_info += '%s | %s\n' % (name, m.training)
    logger.info(layer_info)

    # Print Learnable Parameters
    layer_info = '\n' + '-' * 30 + '\nUnfreezed layer_name\n'
    layer_info += '\n%s\n' % ('Feature Extractor')
    for name, param in visual_encoder.named_parameters():
        if param.requires_grad:
            layer_info += '%s\n' % (name)
    layer_info += '\n%s\n' % ('semantic_encoder')
    for name, param in semantic_encoder.named_parameters():
        if param.requires_grad:
            layer_info += '%s\n' % (name)
    logger.info(layer_info)

    unseen_log = '\nUnseen Classes\n'
    for idx in unseen_classes_idx:
        unseen_log += '%02d - %s\n' % (int(idx), VOC[int(idx)])
    logger.info(unseen_log)

    seen_log = '\nSeen Classes\n'
    for idx in seen_classes_idx:
        seen_log += '%02d - %s\n' % (int(idx), VOC[int(idx)])
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

    trainer = Trainer(RATIO, LAMBDA, THRES,
                      visual_encoder, semantic_encoder,
                      optimizer, optimizer_gen,
                      evaluator,
                      config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      embeddings=dataset.embeddings)
    
    trainer.train()

    message = f"""
    XXXX PASCAL(SPNet) Training CODE finish
    """

    data = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'token': SLACK_TOKEN,
        'channel': CHANNEL_ID,    # User ID. 
        'as_user': True,
        'text': message
    }

    requests.post(url=URL, data=data)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-re', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--ratio', default=1.0, type=float,
                      help='ratio (default: 1.0)')
    args.add_argument('-kl', '--kl', default=0.0, type=float,
                      help='lambda (default: 0.1)')
    args.add_argument('-th', '--thres', default=0.0, type=float,
                      help='threshold (default: 0.0)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--nw', '--num_workers'], type=int, target='data_loader;args;num_workers'),
        CustomArgs(['--n_unseen', '--n_unseen_classes'], type=int, target='data_loader;args;n_unseen_classes')
    ]
    config = ConfigParser.from_args(args, options)

    args_ = args.parse_args()
    main(config, args_.ratio, args_.kl, args_.thres)
    # main(config)
