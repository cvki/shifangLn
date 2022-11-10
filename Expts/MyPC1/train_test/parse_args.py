import torch
import numpy as np
import argparse
from models.LightModel import PCRegLit
from pytorch_lightning import Trainer

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root', default='/d2/code/datas/vs_modelnet40', help='the data directory')
    parser.add_argument('--mode', type=str,
                        help='selcet mode:["train","valid","test"]')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--epoches', type=int, default=60)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=[30, 50],
                        help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='result_models/logs',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=10,
                        help='the frequency to save the logs and checkpoints')
    # parser.add_argument('--train_npts', type=int, default=1024,
    #                     help='the points number of each pc for training')
    # parser.add_argument('--normal', action='store_true',
    #                     help='whether to use normal data')
    # parser.add_argument('--mode', default='clean',
    #                     choices=['clean', 'partial', 'noise'],
    #                     help='training mode about data')
    ## models training
    args = parser.parse_args()
    return args


def train():
    argms=config_params()
    model = PCRegLit(argms)
    trainer = Trainer(
        default_root_dir='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints',
        enable_progress_bar=True, max_epochs=argms.epoches, min_epochs=3, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
