import argparse
import collections
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch

from data_loader import get_seen_idx, get_unseen_idx, VOC
from utils.parse_config import ConfigParser
from pathlib import Path


def main(config):
    logger = config.get_logger('test')
    
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

    val_loader = dataset.get_val_loader()

    # build model architecture, then print to console
    visual_encoder = config.init_obj('arch', module_arch, {"sync_bn": False})
    logger.info(visual_encoder)

    semantic_encoder = config.init_obj('arch_gen', module_arch)
    logger.info(semantic_encoder)

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

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    logger.info(visual_encoder.load_state_dict(checkpoint['state_dict'], strict=False))

    path_fe = str(config.resume)
    path_gen = path_fe.replace('FE', 'Gen')
    path_gen = Path(path_gen)
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
    # embeddings = dataset.embeddings.float().to(device)
    embeddings = torch.load('embeddings/pascal/ZS3/norm_embed_arr_300_airplane.pkl', pickle_allow=True)
    embeddings = embeddings.to(device)
    
    classes_idx = seen_classes_idx + unseen_classes_idx
    classes_idx.sort()

    prototype = semantic_encoder(embeddings)
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
                                torch.index_select(prototype, 0, torch.Tensor(classes_idx).long().to(device)),
                                p=2
                                )  # (N, H, W, 21)
            cdist = cdist**2
            cdist = cdist.permute(0, 3, 1, 2)  # (N, 21, H, W)

            top_k = torch.topk(cdist, k=2, dim=1, largest=False)

            l2 = torch.clone(cdist)  # (N, 20, H, W)
            for idx, class_idx in enumerate(classes_idx):
                if class_idx in unseen_classes_idx:
                    l2[:, idx, :, :] = l2[:, idx, :, :] * config['hyperparameter']['sigma']

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
        log = log + 'thres - %.1f\n' % (config['hyperparameter']['sigma'])
        for met in metric_ftns:
            if 'harmonic' in met().keys():
                log = log + '%s %.1f\n' % (met.__name__ + '_harmonic', met()['harmonic'] * 100)
            if 'seen' in met().keys():
                log = log + '%s %.1f\n' % (met.__name__ + '_seen', met()['seen'] * 100)
            if 'unseen' in met().keys():
                log = log + '%s %.1f\n' % (met.__name__ + '_unseen', met()['unseen'] * 100)
            if 'overall' in met().keys():
                log = log + '%s %.1f\n' % (met.__name__ + '_overall', met()['overall'] * 100)
            if 'by_class' in met().keys():
                log = log + '%s\n' % (met.__name__ + '_by_class')
                for i in classes_idx:
                    if i in unseen_classes_idx:
                        log = log + '%2d *%s %.1f\n' % (i, VOC[i], met()['by_class'][i] * 100)
                    else:
                        log = log + '%2d  %s %.1f\n' % (i, VOC[i], met()['by_class'][i] * 100)
        logger.info(log)

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
        CustomArgs(['--nw', '--num_workers'], type=int, target='data_loader;args;num_workers'),
        CustomArgs(['--s', '--sigma'], type=float, target='hyperparameter;sigma'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
