import argparse
import sys
import time
import logging
import torch
import torch.optim as optim

sys.path.append(r'/d2/code/gitRes/Expts/ptMeta_rgstr')  # import root dir,or running error not find moudle 'xxxx'
from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
import numpy as np
from openpoints.utils import EasyConfig, cal_model_parm_nums
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, cfg_bkb):
        # super().__init__(cfg, cfg_bkb, max_epoch=cfg.optim.max_epoch)
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        # train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        train_loader, val_loader= train_valid_data_loader(cfg, cfg_bkb,self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        # message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        message = 'Calibrate neighbors: {}.'.format('None.ooooooo')
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg,cfg_bkb).cuda()
        model = self.register_model(model)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)


        '''PointMeta change'''
        # # model, optimizer, scheduler
        # # model = create_model(cfg_bkb).cuda()
        # # model = self.register_model(cfg_bkb)
        # if cfg_bkb.model.get('in_channels', None) is None:
        #     cfg_bkb.model.in_channels = cfg_bkb.model.encoder_args.in_channels
        # self.model = build_model_from_cfg(cfg_bkb.model).to(torch.device('cuda'))   # single-gpu
        # self.model_size = cal_model_parm_nums(self.model)
        # # logging.info(self.model)
        # # logging.info('Number of params: %.4f M' % (self.model_size / 1e6))
        # # if cfg_bkb.sync_bn:
        # #    self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # #    logging.info('Using Synchronized BatchNorm ...')
        # # if cfg.distributed:
        # #     torch.cuda.set_device(gpu)
        # #     model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        # #     logging.info('Using Distributed Data parallel ...')
        #
        # # optimizer & scheduler
        # optimizer = build_optimizer_from_cfg(self.model, lr=cfg_bkb.lr, **cfg_bkb.optimizer)
        # scheduler = build_scheduler_from_cfg(cfg_bkb, optimizer)
        # self.register_optimizer(optimizer)
        # self.register_scheduler(scheduler)


        # optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        # self.register_optimizer(optimizer)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        # self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def config_backbone():
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=False,
                        default='/d2/code/gitRes/Expts/ptMeta_rgstr/cfgs/s3dis/pointmetabase-l.yaml',
                        help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg_bkb = EasyConfig()
    cfg_bkb.load(args.cfg, recursive=True)
    cfg_bkb.update(opts)  # overwrite the default arguments in yml

    if cfg_bkb.seed is None:
        cfg_bkb.seed = np.random.randint(1, 10000)

    return cfg_bkb


def main():
    cfg = make_cfg()
    cfg_bkb=config_backbone()
    trainer = Trainer(cfg,cfg_bkb)
    trainer.run()


if __name__ == '__main__':
    main()

