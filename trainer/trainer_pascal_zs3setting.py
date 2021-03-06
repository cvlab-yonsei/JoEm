import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from base import BaseTrainer
from utils import inf_loop, MetricTracker, MetricTracker_scalars
from model.utils import get_lr
from data_loader import VOC, get_unseen_idx


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
        self,
        visual_encoder, semantic_encoder,
        optimizer, optimizer_gen,
        evaluator,
        config,
        train_loader, val_loader=None, test_loader=None,
        lr_scheduler=None, len_epoch=None, embeddings=None,
    ):

        super().__init__(config)

        self.visual_encoder = visual_encoder.to(self.device)
        if len(self.device_ids) > 1:
            self.visual_encoder = torch.nn.DataParallel(visual_encoder, device_ids=self.device_ids)
        self.semantic_encoder = semantic_encoder.to(self.device)
        if len(self.device_ids) > 1:
            self.semantic_encoder = torch.nn.DataParallel(self.semantic_encoder, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.optimizer_gen = optimizer_gen
        
        self.evaluator = evaluator

        self.class_idx = train_loader.dataset.unseen_classes_idx + train_loader.dataset.seen_classes_idx
        self.class_idx.sort()
        self.seen_classes_idx = train_loader.dataset.seen_classes_idx
        self.seen_classes_idx.sort()
        self.unseen_classes_idx = train_loader.dataset.unseen_classes_idx
        self.unseen_classes_idx.sort()

        self.embeddings = embeddings.float().to(self.device)  # [21, 300]

        # we assume semantic feature for void label as zero vector
        self.tmp_zero = torch.zeros_like(self.embeddings[0], device=self.device)
        self.embeddings_cat_zero = torch.cat((self.embeddings, self.tmp_zero.reshape(1, -1)), dim=0)  # [22, 300]

        self.train_loader = train_loader
        if len_epoch is None:  # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:  # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch

        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.test_loader = test_loader

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_loader.batch_size))

        self.metric_ftns = [getattr(self.evaluator, met) for met in config['metrics']]
        self.train_metrics = MetricTracker(
            'loss',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        
        self.change_label = False
        if config['arch']['args']['num_classes'] != 21:
            self.change_label = True

        self.RATIO = config['hyperparameter']['ratio']
        self.TEMP = config['hyperparameter']['temperature']
        self.ALPHA = config['hyperparameter']['alpha']
        self.SIGMA = config['hyperparameter']['sigma']

        self.logger.info('-' * 30)
        self.logger.info('BAR Loss')
        self.logger.info('  **BAR : %.4f**' % (self.RATIO))

        self.logger.info('SC Loss')
        self.logger.info('  **Temp  : %d**' % (self.TEMP))

        self.logger.info('ALPHA: %.4f**' % (self.ALPHA))
        self.logger.info('-' * 30)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _train_epoch(self, epoch):

        self.visual_encoder.train()
        if isinstance(self.visual_encoder, nn.DataParallel):
            self.visual_encoder.module.freeze_bn()
        else:
            self.visual_encoder.freeze_bn()
        self.semantic_encoder.train()

        self.train_metrics.reset()
        for batch_idx, data in enumerate(tqdm.tqdm(self.train_loader)):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)

            self.optimizer.zero_grad()
            self.optimizer_gen.zero_grad()
            
            logit, real_feature = self.visual_encoder(data['image'])
            N, C, h, w = real_feature.shape

            # ============
            #  BAR Loss
            # ============
            # 'Nearest' Downscale
            target_downscale = F.interpolate(
                data['label'].unsqueeze(0),
                size=(int(h * self.RATIO), int(w * self.RATIO)),
                mode="nearest",
            ).squeeze(0)  # [N, h, w]

            # fill semantic feature into downscaled lable map
            # fill zero-vector, if it is ignore labels
            target_downscale[target_downscale == 255] = len(self.class_idx)
            embeddings_input = torch.index_select(
                self.embeddings_cat_zero,
                dim=0,
                index=target_downscale.flatten().long()
            ).reshape(N, int(h * self.RATIO), int(w * self.RATIO), -1).permute(0, 3, 1, 2)  # [N, 600, h', w']
            
            # 'Bilinear' Upscaling
            embeddings_input = F.interpolate(
                embeddings_input,
                size=(h, w),
                mode="bilinear", align_corners=False
            )
            
            # forward pass, semantic_encoder
            proto = self.semantic_encoder(embeddings_input.permute(0, 2, 3, 1).detach())  # [N, H, W, 600]
            real_feature = real_feature.permute(0, 2, 3, 1)  # [N, H, W, 300]

            target_resize = F.interpolate(
                data['label'].unsqueeze(0),
                size=(h, w),
                mode="nearest",
            ).squeeze(0)  # [N, h, w]

            # Center Loss
            loss_gen = ((proto[target_resize != 255] - real_feature[target_resize != 255])**2).mean()

            # ===========
            #  CE Loss
            # ===========
            label = data['label'].clone().detach()
            if self.change_label:
                for idx in self.class_idx:
                    label[data['label'] == idx] = self.class_idx.index(idx)
            loss_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')(logit, label.long())

            # ===========
            #  SC Loss
            # ===========
            idx_list = self.seen_classes_idx[:]
            if 0 in idx_list:
                idx_list.remove(0)  # Only impose SC loss on object
            semantic_relation_l2 = torch.cdist(
                torch.index_select(self.embeddings, 0, torch.Tensor(idx_list).long().to(self.device)),
                torch.index_select(self.embeddings, 0, torch.Tensor(idx_list).long().to(self.device)),
                p=2
            )
            semantic_relation = F.softmax(-semantic_relation_l2 * self.TEMP, dim=-1)  # [15, 15]

            proto_embeddings = self.semantic_encoder(self.embeddings.detach())  # [21, 256]

            proto_relation_l2 = torch.cdist(
                torch.index_select(proto_embeddings, 0, torch.Tensor(idx_list).long().to(self.device)),
                torch.index_select(proto_embeddings, 0, torch.Tensor(idx_list).long().to(self.device)),
                p=2
            )
            proto_relation = F.log_softmax(-proto_relation_l2, dim=-1)  # [15, 15]

            loss_sc = nn.KLDivLoss(reduction='batchmean')(proto_relation, semantic_relation)  # input:log, target:prob.

            loss = loss_gen + loss_ce + self.ALPHA * loss_sc
            loss.backward()

            self.optimizer.step()
            self.optimizer_gen.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars(
                    'lr',
                    {'lr_CE': get_lr(self.optimizer)[0],
                     'lr_Gen': get_lr(self.optimizer_gen)[0]},
                    epoch - 1
                )

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            val_flag = True

        if self.lr_scheduler is not None:
            self.lr_scheduler(self.optimizer, 0, epoch)

        return log, val_flag

    def _valid_epoch(self, epoch):
        self.visual_encoder.eval()
        self.semantic_encoder.eval()
        prototype = self.semantic_encoder(self.embeddings)  # [21, 300]

        log = {}
        self.evaluator.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm.tqdm(self.val_loader)):

                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                _, real_feature = self.visual_encoder(data['image'])
                N, C, h, w = real_feature.shape

                real_feature = F.interpolate(real_feature, size=data['image'].size()[2:], mode="bilinear", align_corners=False)

                real_feature = real_feature.permute(0, 2, 3, 1)

                cdist = torch.cdist(real_feature,
                                    torch.index_select(prototype, 0, torch.Tensor(self.class_idx).long().to(self.device)),
                                    p=2
                                    )  # (N, H, W, 21)
                cdist = cdist**2
                cdist = cdist.permute(0, 3, 1, 2)  # (N, 21, H, W)

                top_k = torch.topk(cdist, k=2, dim=1, largest=False)

                l2 = torch.clone(cdist)  # (N, 20, H, W)
                for idx, class_idx in enumerate(self.class_idx):
                    if class_idx in self.unseen_classes_idx:
                        l2[:, idx, :, :] = l2[:, idx, :, :] * self.SIGMA  # Threshold

                pred = torch.clone(top_k.indices[:, 0, :, :])  # [N, H, W]
                l2_min = l2.gather(1, (top_k.indices[:, 0, :, :]).unsqueeze(1)).squeeze(1)  # [N, H, W]
                for i in range(1):
                    mask = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1) < l2_min
                    l2_min[mask] = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1)[mask]
                    pred[mask] = top_k.indices[:, i + 1, :, :][mask]

                pred = pred.cpu().numpy()

                self.evaluator.add_batch(target, pred)

            self.writer.set_step((epoch), 'valid')
            
            for met in self.metric_ftns:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['seen'], met()['unseen'], met()['harmonic']], 'seen', 'unseen', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': met()['harmonic']})
                if 'seen' in met().keys():
                    log.update({met.__name__ + '_seen': met()['seen']})
                if 'unseen' in met().keys():
                    log.update({met.__name__ + '_unseen': met()['unseen']})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': met()['overall']})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in get_unseen_idx(self.config['data_loader']['args']['n_unseen_classes']):
                            by_class_str = by_class_str + '%2d *%s %.3f\n' % (i, VOC[i], met()['by_class'][i])
                        else:
                            by_class_str = by_class_str + '%2d  %s %.3f\n' % (i, VOC[i], met()['by_class'][i])
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _test(self):
        self.logger.info("TEST")

        self.visual_encoder.eval()
        self.semantic_encoder.eval()
        prototype = self.semantic_encoder(self.embeddings)  # [21, 300]

        log = {}
        self.evaluator.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm.tqdm(self.test_loader)):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                _, real_feature = self.visual_encoder(data['image'])
                N, C, h, w = real_feature.shape

                real_feature = F.interpolate(real_feature, size=data['image'].size()[2:], mode="bilinear", align_corners=False)

                real_feature = real_feature.permute(0, 2, 3, 1)

                cdist = torch.cdist(
                    real_feature,
                    torch.index_select(prototype, 0, torch.Tensor(self.class_idx).long().to(self.device)),
                    p=2
                )  # (N, H, W, 21)
                cdist = cdist**2
                cdist = cdist.permute(0, 3, 1, 2)  # (N, 21, H, W)

                top_k = torch.topk(cdist, k=2, dim=1, largest=False)

                l2 = torch.clone(cdist)  # (N, 20, H, W)
                for idx, class_idx in enumerate(self.class_idx):
                    if class_idx in self.unseen_classes_idx:
                        l2[:, idx, :, :] = l2[:, idx, :, :] * self.SIGMA  # Threshold

                pred = torch.clone(top_k.indices[:, 0, :, :])  # [N, H, W]
                l2_min = l2.gather(1, (top_k.indices[:, 0, :, :]).unsqueeze(1)).squeeze(1)  # [N, H, W]
                for i in range(1):
                    mask = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1) < l2_min
                    l2_min[mask] = l2.gather(1, (top_k.indices[:, i + 1, :, :]).unsqueeze(1)).squeeze(1)[mask]
                    pred[mask] = top_k.indices[:, i + 1, :, :][mask]

                pred = pred.cpu().numpy()

                self.evaluator.add_batch(target, pred)
            
            for met in self.metric_ftns:
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': met()['harmonic']})
                if 'seen' in met().keys():
                    log.update({met.__name__ + '_seen': met()['seen']})
                if 'unseen' in met().keys():
                    log.update({met.__name__ + '_unseen': met()['unseen']})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': met()['overall']})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in get_unseen_idx(self.config['data_loader']['args']['n_unseen_classes']):
                            by_class_str = by_class_str + '%2d *%s %.3f\n' % (i, VOC[i], met()['by_class'][i])
                        else:
                            by_class_str = by_class_str + '%2d  %s %.3f\n' % (i, VOC[i], met()['by_class'][i])
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.visual_encoder).__name__
        if isinstance(self.visual_encoder, nn.DataParallel):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.visual_encoder.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.visual_encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = str(self.checkpoint_dir / 'fe_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)

        arch = type(self.semantic_encoder).__name__
        if isinstance(self.semantic_encoder, nn.DataParallel):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.semantic_encoder.module.state_dict(),
                'optimizer': self.optimizer_gen.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.semantic_encoder.state_dict(),
                'optimizer': self.optimizer_gen.state_dict(),
                'monitor_best': self.mnt_best,
            }
        
        filename = str(self.checkpoint_dir / 'gen_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)

        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_model(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.visual_encoder).__name__
        if isinstance(self.visual_encoder, nn.DataParallel):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.visual_encoder.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.visual_encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
            }
        best_path = str(self.checkpoint_dir / 'model_best_fe.pth')
        torch.save(state, best_path)

        arch = type(self.semantic_encoder).__name__
        if isinstance(self.semantic_encoder, nn.DataParallel):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.semantic_encoder.module.state_dict(),
                'optimizer': self.optimizer_gen.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.semantic_encoder.state_dict(),
                'optimizer': self.optimizer_gen.state_dict(),
                'monitor_best': self.mnt_best,
            }
        best_path = str(self.checkpoint_dir / 'model_best_gen.pth')
        torch.save(state, best_path)

        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        path_fe = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(path_fe))
        checkpoint = torch.load(path_fe)
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

        if not self.reset_best_mnt:
            if 'monitor_best' in checkpoint:
                self.mnt_best = checkpoint['monitor_best']
                self.logger.info('Monitor Best: %.4f' % (self.mnt_best))
            
        if len(self.device_ids) > 1:
            self.logger.info(self.visual_encoder.module.load_state_dict(checkpoint['state_dict']))
        else:
            self.visual_encoder.load_state_dict(checkpoint['state_dict'])
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        path_gen = path_fe.replace('fe', 'gen')
        self.logger.info("Loading checkpoint: {} ...".format(path_gen))
        path_gen = Path(path_gen)
        checkpoint = torch.load(path_gen)
            
        if len(self.device_ids) > 1:
            self.logger.info(self.semantic_encoder.module.load_state_dict(checkpoint['state_dict']))
        else:
            self.logger.info(self.semantic_encoder.load_state_dict(checkpoint['state_dict']))

        if 'optimizer' in checkpoint:
            self.optimizer_gen.load_state_dict(checkpoint['optimizer'])
