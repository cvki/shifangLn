import open3d
import argparse
from cgi import test
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ModelNet40 import MyModelNet40
from models import IterativeBenchmark
from loss import EMDLosspy
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root', default='/d2/code/datas/MyModelNet40', help='the data path')
    # parser.add_argument('--train_npts', type=int, default=1024,
    #                     help='the points number of each pc for training')
    parser.add_argument('--normal', action='store_true',
                        help='whether to use normal data')
    parser.add_argument('--mode', default='clean',
                        choices=['clean', 'partial', 'noise'],
                        help='training mode about data')
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--epoches', type=int, default=60)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=[30, 50],
                        help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=10,
                        help='the frequency to save the logs and checkpoints')
    args = parser.parse_args()
    return args


def compute_loss(ref_cloud, pred_ref_clouds, loss_fn):
    losses = []
    discount_factor = 0.5
    for i in range(8):
        loss = loss_fn(ref_cloud[..., :3].contiguous(),
                       pred_ref_clouds[i][..., :3].contiguous())
        losses.append(discount_factor**(8 - i)*loss)
    return torch.sum(torch.stack(losses))


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for ref_cloud, src_cloud, gtR, gtt in tqdm(train_loader):
        ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), gtR.cuda(), gtt.cuda()
        optimizer.zero_grad()
        R, t, pred_ref_clouds = model(src_cloud.permute(0, 2, 1).contiguous(), ref_cloud.permute(0, 2, 1).contiguous())
        loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
        loss.backward()
        optimizer.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for ref_cloud, src_cloud, gtR, gtt in tqdm(test_loader):
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), gtR.cuda(), gtt.cuda()
            R, t, pred_ref_clouds = model(src_cloud.permute(0, 2, 1).contiguous(), ref_cloud.permute(0, 2, 1).contiguous())
            loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropic = compute_metrics(R, t, gtR, gtt)

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params()
    print(args)

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    train_set = MyModelNet40(root=args.root, train=True, normal=args.normal, mode=args.mode, size=6000)
    test_set = MyModelNet40(root=args.root, train=False, normal=args.normal, mode=args.mode, size=600)
    
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    # Init our model
    from models.benchmark import IterativeBenchmarkLightning
    mnist_model = IterativeBenchmarkLightning()

    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader, test_loader)

if __name__ == '__main__':
    main()