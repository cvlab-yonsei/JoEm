import argparse
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger.logger import setup_logging
from utils.utils import read_json, write_json

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from data_loader import get_seen_idx, get_unseen_idx, CONTEXT_59
from utils import MetricTracker
from pathlib import Path

import json
import requests
from pandas.io.json import json_normalize

URL         = 'https://slack.com/api/chat.postMessage'
SLACK_TOKEN = 'xoxb-116215949047-1731607340946-BNe4GC18rHG6I6ioY10r00vy'
CHANNEL_ID  = 'UKDHE8QLV'

def main(config, thres):
    logger = config.get_logger('test')
    logger.info('**Resume %s **' % (str(config.resume)))

    # thres = 0.4
    logger.info('**Model best - thres %.1f**' % (thres))

    # setup data_loader instances
    n_class = config['data_loader']['args']['n_unseen_classes']
    unseen_classes_idx = get_unseen_idx(n_class, dataset='context59')
    ignore_bg = config['data_loader']['args']['train']['args']['ignore_bg']
    seen_classes_idx = get_seen_idx(n_class, ignore_bg=ignore_bg, dataset='context59')
    info = {"unseen_classes_idx": unseen_classes_idx,
            "seen_classes_idx": seen_classes_idx}
    dataset = config.init_obj('data_loader', module_data, **info)

    if dataset.val_set.ignore_bg:
        setting = 'SPNet'
    else:
        setting = 'ZS3Net'
    logger.info('Number of images in val(%s): %d' % (setting, len(dataset.val_set)))

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

    unseen_log = '\nUnseen Classes\n'
    for idx in unseen_classes_idx:
        unseen_log += '%02d - %s\n' % (int(idx), CONTEXT_59[int(idx)])
    logger.info(unseen_log)

    seen_log = '\nSeen Classes\n'
    for idx in seen_classes_idx:
        seen_log += '%02d - %s\n' % (int(idx), CONTEXT_59[int(idx)])
    logger.info(seen_log)

    info = [dataset.num_classes, seen_classes_idx, unseen_classes_idx]
    evaluator = config.init_obj('evaluator', module_metric, *info)

    best_epoch = 0
    best_harmonic = 0.
    for epoch in range(1, config['trainer']['epochs']+1):
        path_fe = config.resume / ('FE_checkpoint-epoch%d.pth' % (epoch))
        logger.info('Loading checkpoint: {} ...'.format(path_fe))
        checkpoint = torch.load(path_fe)
        logger.info(visual_encoder.load_state_dict(checkpoint['state_dict'], strict=False))

        path_gen = config.resume / ('Gen_checkpoint-epoch%d.pth' % (epoch))
        logger.info('Loading checkpoint: {} ...'.format(path_gen))
        checkpoint = torch.load(path_gen)
        logger.info(semantic_encoder.load_state_dict(checkpoint['state_dict'], strict=False))
        
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        visual_encoder = visual_encoder.to(device)
        semantic_encoder = semantic_encoder.to(device)
    
        # total_loss = 0.0
        metric_ftns = [getattr(evaluator, met) for met in config['metrics']]
        
        visual_encoder.eval()
        semantic_encoder.eval()
        embeddings = dataset.embeddings.float().to(device)
        
        seen_classes_idx = val_loader.dataset.seen_classes_idx
        unseen_classes_idx = val_loader.dataset.unseen_classes_idx
        classes_idx = seen_classes_idx + unseen_classes_idx
        classes_idx.sort()

        w2v = semantic_encoder(embeddings)
    
        with torch.no_grad():
            evaluator.reset()
            for batch_idx, data in enumerate(tqdm.tqdm(val_loader)):
                data['image'], data['label'] = data['image'].to(device), data['label'].to(device)
                target = data['label'].cpu().numpy()

                _, real_feature = visual_encoder(data['image'])
                N, C, h, w = real_feature.shape

                real_feature = F.interpolate(real_feature, size=data['image'].size()[2:], mode="bilinear", align_corners=False)

                real_feature = real_feature.permute(0, 2, 3, 1)

                cdist = torch.cdist(real_feature,
                                    torch.index_select(w2v, 0, torch.Tensor(classes_idx).long().to(device)),
                                    p=2
                                    )  # (N, H, W, 21)
                cdist = cdist**2
                cdist = cdist.permute(0, 3, 1, 2)  # (N, 21, H, W)

                top_k = torch.topk(cdist, k=2, dim=1, largest=False)

                l2 = torch.clone(cdist)  # (N, 20, H, W)
                for idx, class_idx in enumerate(classes_idx):
                    if class_idx in unseen_classes_idx:
                        l2[:, idx, :, :] = l2[:, idx, :, :] * thres

                pred = torch.clone(top_k.indices[:, 0, :, :])  # [N, H, W]
                l2_min = l2.gather(1, (top_k.indices[:, 0, :, :]).unsqueeze(1)).squeeze(1)  # [N, H, W]
                for i in range(1):
                    mask = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1) < l2_min
                    l2_min[mask] = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1)[mask]
                    pred[mask] = top_k.indices[:, i + 1, :, :][mask]

                pred_ = torch.clone(pred)
                for uni_cls in pred.unique():
                    pred_[pred == uni_cls] = classes_idx[uni_cls.long()]

                pred_ = pred_.cpu().numpy()
                evaluator.add_batch(target, pred_)

            log = '\n'
            log = log + 'thres - %.1f\n' % (thres)
            for met in metric_ftns:
                if 'harmonic' in met().keys():
                    log = log + '%s %.2f\n' % (met.__name__ + '_harmonic', met()['harmonic']*100)
                if 'seen' in met().keys():
                    log = log + '%s %.2f\n' % (met.__name__ + '_seen', met()['seen']*100)
                if 'unseen' in met().keys():
                    log = log + '%s %.2f\n' % (met.__name__ + '_unseen', met()['unseen']*100)
                if 'overall' in met().keys():
                    log = log + '%s %.2f\n' % (met.__name__ + '_overall', met()['overall']*100)
                if 'by_class' in met().keys():
                    log = log + '%s\n' % (met.__name__+'_by_class')
                    for i in classes_idx:
                        if i in unseen_classes_idx:
                            log = log + '%2d *%s %.2f\n' % (i, CONTEXT_59[i], met()['by_class'][i]*100)
                        else:
                            log = log + '%2d  %s %.2f\n' % (i, CONTEXT_59[i], met()['by_class'][i]*100)
            logger.info(log)

            for met in metric_ftns:
                if 'Mean_Intersection_over_Union' == met.__name__:
                    if best_harmonic < met()['harmonic']:
                        best_epoch = epoch
                        best_harmonic = met()['harmonic']
                        logger.info('model_best!')

    # print best model epoch and performance
    logger.info('-'*25)
    logger.info('Best Model - Epoch: %d' % (best_epoch))
    logger.info('MIoU harmonic - %.2f' % (best_harmonic*100))

    message = f"""

    XXX TEST CODE finish
    Train Context ZS3 setting
    """

    data = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'token': SLACK_TOKEN,
        'channel': CHANNEL_ID,    # User ID. 
        'as_user': True,
        'text': message
    }

    requests.post(url=URL, data=data)
    

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        
    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            # cfg_fname = resume.parent / 'config.json'
            cfg_fname = resume / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))
            
        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        
        try:
            return getattr(module, module_name)(*args, **module_args)
        except AttributeError:
            return None
        

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-th', '--thres', default=1.0, type=float,
                      help='Threshold (default: 1.0)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n_gpu', '--number_of_gpu'], type=int, target='n_gpu'),
        CustomArgs(['--cs', '--number_of_classifier_class'], type=int, target='arch;args;num_classes'),
        CustomArgs(['--pt', '--pretrained'], type=bool, target='arch;args;pretrained'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--nw', '--num_workers'], type=int, target='data_loader;args;num_workers')
    ]
    config = ConfigParser.from_args(args, options)

    args_ = args.parse_args()
    main(config, args_.thres)
