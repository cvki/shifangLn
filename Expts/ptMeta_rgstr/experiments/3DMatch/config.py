import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir

import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
# from tqdm import tqdm
# import torch, torch.nn as nn
# from torch import distributed as dist, multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
# from torch_scatter import scatter
# from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
#     cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
# from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
# from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
# from openpoints.dataset.data_util import voxelize
# from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
# from openpoints.transforms import build_transforms_from_cfg
# from openpoints.optim import build_optimizer_from_cfg
# from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
# from openpoints.models import build_model_from_cfg
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, 'data', '3DMatch')

# train data
_C.train = edict()
_C.train.batch_size = 1
# _C.train.num_workers = 8
_C.train.num_workers = 2
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 0
# _C.test.num_workers = 8
_C.test.point_limit = None

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 40
_C.optim.grad_acc_steps = 1

# # PointMeta optim:
# NAME: 'adamw'  # performs 1 point better than adam
# weight_decay: 1.0e-4

# # model - backbone
# _C.backbone = edict()
# _C.backbone.NAME='BaseSeg'
#
# _C.backbone.encoder_args=edict()
# _C.backbone.encoder_args.NAME='PointMetaBaseEncoder'
# _C.backbone.encoder_args.blocks= [1, 3, 5, 3, 3] #[1, 4, 7, 4, 4] #[1, 3, 5, 3, 3]
# _C.backbone.encoder_args.strides= [1, 4, 4, 4, 4]
# _C.backbone.encoder_args.sa_layers= 1
# _C.backbone.encoder_args.sa_use_res= False
# _C.backbone.encoder_args.width= 32
# _C.backbone.encoder_args.in_channels= 4
# _C.backbone.encoder_args.expansion= 1 #4
# _C.backbone.encoder_args.radius= 0.1
# _C.backbone.encoder_args.nsample= 32
# _C.backbone.encoder_args.aggr_args=edict()
# _C.backbone.encoder_args.aggr_args.feature_type= 'dp_fj'
# _C.backbone.encoder_args.aggr_args.reduction= 'max'
# _C.backbone.encoder_args.group_args=edict()
# _C.backbone.encoder_args.group_args.NAME= 'ballquery'
# _C.backbone.encoder_args.group_args.normalize_dp= True
# _C.backbone.encoder_args.conv_args=edict()
# _C.backbone.encoder_args.conv_argsorder= 'conv-norm-act'
# _C.backbone.encoder_args.act_args=edict()
# _C.backbone.encoder_args.act_args.act= 'relu'
# _C.backbone.encoder_args.norm_args=edict()
# _C.backbone.encoder_args.norm_args.norm= 'bn'
#
# _C.backbone.decoder_args=edict()
# _C.backbone.decoder_args.NAME= 'PointNextDecoder'
#
# _C.backbone.cls_args=edict()
# _C.backbone.cls_args.NAME= 'SegHead'
# _C.backbone.cls_args.num_classes= 13
# _C.backbone.cls_args.in_channels= 'null'
# _C.backbone.cls_args.norm_args=edict()
# _C.backbone.cls_args.norm_args.norm= 'bn'
#
# batch_size= 8
# seed= 2425 #1111 #4333 #2425

# _C.backbone = edict()
# _C.backbone.num_stages = 4
# _C.backbone.init_voxel_size = 0.025
# _C.backbone.kernel_size = 15
# _C.backbone.base_radius = 2.5
# _C.backbone.base_sigma = 2.0
# _C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
# _C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
# _C.backbone.group_norm = 32
# _C.backbone.input_dim = 1
# _C.backbone.init_dim = 64
# _C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
# _C.geotransformer.input_dim = 1024
_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args

#
# def modelx(gpu,cfg):
#
#     # if cfg.distributed:
#     #     if cfg.mp:
#     #         cfg.rank = gpu
#     #     dist.init_process_group(backend=cfg.dist_backend,
#     #                             init_method=cfg.dist_url,
#     #                             world_size=cfg.world_size,
#     #                             rank=cfg.rank)
#     #     dist.barrier()
#     #
#     # # logger
#     # setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
#     # if cfg.rank == 0:
#     #     Wandb.launch(cfg, cfg.wandb.use_wandb)
#     #     writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
#     # else:
#     #     writer = None
#     # set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
#     # torch.backends.cudnn.enabled = True
#     # logging.info(cfg)
#
#     if cfg.model.get('in_channels', None) is None:
#         cfg.model.in_channels = cfg.model.encoder_args.in_channels
#     model = build_model_from_cfg(cfg.model).to(cfg.rank)
#     model_size = cal_model_parm_nums(model)
#     logging.info(model)
#     logging.info('Number of params: %.4f M' % (model_size / 1e6))
#
#     if cfg.sync_bn:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         logging.info('Using Synchronized BatchNorm ...')
#     if cfg.distributed:
#         torch.cuda.set_device(gpu)
#         model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
#         logging.info('Using Distributed Data parallel ...')
#
#     # optimizer & scheduler
#     optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
#     scheduler = build_scheduler_from_cfg(cfg, optimizer)
#
#     # build dataset
#     # val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
#     #                                        cfg.dataset,
#     #                                        cfg.dataloader,
#     #                                        datatransforms_cfg=cfg.datatransforms,
#     #                                        split='val',
#     #                                        distributed=cfg.distributed
#     #                                        )
#     # logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
#     # num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
#     # if num_classes is not None:
#     #     assert cfg.num_classes == num_classes
#     # logging.info(f"number of classes of the dataset: {num_classes}")
#     # cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
#     # cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
#     # validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere
#     #
#     # # optionally resume from a checkpoint
#     # model_module = model.module if hasattr(model, 'module') else model
#     # if cfg.pretrained_path is not None:
#     #     if cfg.mode == 'resume':
#     #         resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
#     #     else:
#     #         if cfg.mode == 'val':
#     #             best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
#     #             val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1)
#     #             with np.printoptions(precision=2, suppress=True):
#     #                 logging.info(
#     #                     f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
#     #                     f'\niou per cls is: {val_ious}')
#     #             return val_miou
#     #         elif cfg.mode == 'test':
#     #             best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
#     #             data_list = generate_data_list(cfg)
#     #             logging.info(f"length of test dataset: {len(data_list)}")
#     #             test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)
#     #
#     #             if test_miou is not None:
#     #                 with np.printoptions(precision=2, suppress=True):
#     #                     logging.info(
#     #                         f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
#     #                         f'\niou per cls is: {test_ious}')
#     #                 cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
#     #                 write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
#     #             return test_miou
#     #
#     #         elif 'encoder' in cfg.mode:
#     #             logging.info(f'Finetuning from {cfg.pretrained_path}')
#     #             load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
#     #         else:
#     #             logging.info(f'Finetuning from {cfg.pretrained_path}')
#     #             load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
#     # else:
#     #     logging.info('Training from scratch')
#     # if 'freeze_blocks' in cfg.mode:
#     #     for p in model_module.encoder.blocks.parameters():
#     #         p.requires_grad = False
#     #
#     # train_loader = build_dataloader_from_cfg(cfg.batch_size,
#     #                                          cfg.dataset,
#     #                                          cfg.dataloader,
#     #                                          datatransforms_cfg=cfg.datatransforms,
#     #                                          split='train',
#     #                                          distributed=cfg.distributed,
#     #                                          )
#     # logging.info(f"length of training dataset: {len(train_loader.dataset)}")
#     # cfg.criterion_args.weight = None
#     # if cfg.get('cls_weighed_loss', False):
#     #     if hasattr(train_loader.dataset, 'num_per_class'):
#     #         cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
#     #     else:
#     #         logging.info('`num_per_class` attribute is not founded in dataset')
#     # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
#     #
#     # # ===> start training
#     # if cfg.use_amp:
#     #     scaler = torch.cuda.amp.GradScaler()
#     # else:
#     #     scaler = None
#     #
#     # val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
#     # best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
#     # for epoch in range(cfg.start_epoch, cfg.epochs + 1):
#     #     if cfg.distributed:
#     #         train_loader.sampler.set_epoch(epoch)
#     #     if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
#     #         train_loader.dataset.epoch = epoch - 1
#     #     train_loss, train_miou, train_macc, train_oa, _, _ = \
#     #         train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)
#     #
#     #     is_best = False
#     #     if epoch % cfg.val_freq == 0:
#     #         val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
#     #         if val_miou > best_val:
#     #             is_best = True
#     #             best_val = val_miou
#     #             macc_when_best = val_macc
#     #             oa_when_best = val_oa
#     #             ious_when_best = val_ious
#     #             best_epoch = epoch
#     #             with np.printoptions(precision=2, suppress=True):
#     #                 logging.info(
#     #                     f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
#     #                     f'\nmious: {val_ious}')
#     #
#     #     lr = optimizer.param_groups[0]['lr']
#     #     logging.info(f'Epoch {epoch} LR {lr:.6f} '
#     #                  f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
#     #     if writer is not None:
#     #         writer.add_scalar('best_val', best_val, epoch)
#     #         writer.add_scalar('val_miou', val_miou, epoch)
#     #         writer.add_scalar('macc_when_best', macc_when_best, epoch)
#     #         writer.add_scalar('oa_when_best', oa_when_best, epoch)
#     #         writer.add_scalar('val_macc', val_macc, epoch)
#     #         writer.add_scalar('val_oa', val_oa, epoch)
#     #         writer.add_scalar('train_loss', train_loss, epoch)
#     #         writer.add_scalar('train_miou', train_miou, epoch)
#     #         writer.add_scalar('train_macc', train_macc, epoch)
#     #         writer.add_scalar('lr', lr, epoch)
#     #
#     #     if cfg.sched_on_epoch:
#     #         scheduler.step(epoch)
#     #     if cfg.rank == 0:
#     #         save_checkpoint(cfg, model, epoch, optimizer, scheduler,
#     #                         additioanl_dict={'best_val': best_val},
#     #                         is_best=is_best
#     #                         )
#     #         is_best = False
#     # # do not save file to wandb to save wandb space
#     # # if writer is not None:
#     # #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
#     # # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
#     #
#     # # validate
#     # with np.printoptions(precision=2, suppress=True):
#     #     logging.info(
#     #         f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
#     #         f'\niou per cls is: {ious_when_best}')
#     #
#     # if cfg.world_size < 2:  # do not support multi gpu testing
#     #     # test
#     #     load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
#     #     cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'.csv')
#     #     if 'sphere' in cfg.dataset.common.NAME.lower():
#     #         test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg)
#     #     else:
#     #         data_list = generate_data_list(cfg)
#     #         test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)
#     #     with np.printoptions(precision=2, suppress=True):
#     #         logging.info(
#     #             f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
#     #             f'\niou per cls is: {test_ious}')
#     #     if writer is not None:
#     #         writer.add_scalar('test_miou', test_miou, epoch)
#     #         writer.add_scalar('test_macc', test_macc, epoch)
#     #         writer.add_scalar('test_oa', test_oa, epoch)
#     #     write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
#     #     logging.info(f'save results in {cfg.csv_path}')
#     #     if cfg.use_voting:
#     #         load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
#     #         set_random_seed(cfg.seed)
#     #         val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
#     #                                                                      data_transform=data_transform)
#     #         if writer is not None:
#     #             writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)
#     #
#     #         ious_table = [f'{item:.2f}' for item in val_ious]
#     #         data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}',
#     #                 f'{val_miou:.2f}'] + ious_table + [
#     #                    str(best_epoch), cfg.run_dir]
#     #         with open(cfg.csv_path, 'w', encoding='UT8') as f:
#     #             writer = csv.writer(f)
#     #             writer.writerow(data)
#     # else:
#     #     logging.warning(
#     #         'Testing using multiple GPUs is not allowed for now. Running testing after this training is required.')
#     # if writer is not None:
#     #     writer.close()
#     # dist.destroy_process_group()
#     # wandb.finish(exit_code=True)
#

def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')

    # parser = argparse.ArgumentParser('Scene segmentation training/testing')
    # parser.add_argument('--cfg', type=str, required=True, help='config file')
    # parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    # args, opts = parser.parse_known_args()
    # cfg = EasyConfig()
    # cfg.load(args.cfg, recursive=True)
    # cfg.update(opts)  # overwrite the default arguments in yml
    #
    # if cfg.seed is None:
    #     cfg.seed = np.random.randint(1, 10000)
    #
    # # init distributed env first, since logger depends on the dist info.
    # cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    # cfg.sync_bn = cfg.world_size > 1
    #
    # # init log dir
    # cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    # cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    # tags = [
    #     cfg.task_name,  # task name (the folder of name under ./cfgs
    #     cfg.mode,
    #     cfg.cfg_basename,  # cfg file name
    #     f'ngpus{cfg.world_size}',
    #     f'seed{cfg.seed}',
    # ]
    # for i, opt in enumerate(opts):
    #     if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
    #         tags.append(opt)
    # cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    #
    # cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    # if cfg.mode in ['resume', 'val', 'test']:
    #     resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    #     cfg.wandb.tags = [cfg.mode]
    # else:
    #     generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    #     cfg.wandb.tags = tags
    # os.environ["JOB_LOG_DIR"] = cfg.log_dir
    # cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    # with open(cfg_path, 'w') as f:
    #     yaml.dump(cfg, f, indent=2)
    #     os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    # cfg.cfg_path = cfg_path
    #
    # # wandb config
    # cfg.wandb.name = cfg.run_name
    #
    # # multi processing.
    # if cfg.mp:
    #     port = find_free_port()
    #     cfg.dist_url = f"tcp://localhost:{port}"
    #     print('using mp spawn for distributed training')
    #     mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    # else:
    #     modelx(0, cfg)
    #

if __name__ == '__main__':
    main()
