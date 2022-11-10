# import torch
# import argparse
# from lib.utils import load_config
# from easydict import EasyDict as edict
# import os, json, shutil
# from configs.models import architectures
# from torch import optim
# from models.architectures import KPFCNN
#
#
# if __name__=='__main__':
#     print(torch.cuda.is_available())
#     x = torch.rand(5, 3)
#     print(x)
#
#     parser = argparse.ArgumentParser()
#     # add_argument: arg_pos1: "xxx, -xxx, --xxx, ------xxx", differ from "xxx" and "-...xxx"
#     parser.add_argument('config', type=str, help= 'Path to the config file.')
#     args = parser.parse_args()
#     print(args)
#
#
#     # load configs
#     config = load_config(args.config)
#     config['snapshot_dir'] = 'testAPI/%s' % config['exp_dir']
#     config['tboard_dir'] = 'testAPI/%s/tensorboard' % config['exp_dir']
#     config['save_dir'] = 'testAPI/%s/checkpoints' % config['exp_dir']
#     config = edict(config)
#
#     os.makedirs(config.snapshot_dir, exist_ok=True)
#     os.makedirs(config.save_dir, exist_ok=True)
#     os.makedirs(config.tboard_dir, exist_ok=True)
#     json.dump(
#         config,
#         open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
#         indent=4,
#     )
#     if config.gpu_mode:
#         config.device = torch.device('cuda')
#     else:
#         config.device = torch.device('cpu')
#
#     # backup the files
#     os.system(f'cp -r models {config.snapshot_dir}')
#     os.system(f'cp -r datasets {config.snapshot_dir}')
#     os.system(f'cp -r lib {config.snapshot_dir}')
#     shutil.copy2('main.py', config.snapshot_dir)
#
#     # model initialization
#     config.architecture = architectures[config.dataset]
#     config.model = KPFCNN(config)
#
#     # create optimizer
#     if config.optimizer == 'SGD':
#         config.optimizer = optim.SGD(
#             config.model.parameters(),
#             lr=config.lr,
#             momentum=config.momentum,
#             weight_decay=config.weight_decay,
#         )
#     elif config.optimizer == 'ADAM':
#         config.optimizer = optim.Adam(
#             config.model.parameters(),
#             lr=config.lr,
#             betas=(0.9, 0.999),
#             weight_decay=config.weight_decay,
#         )
#
#     # create learning rate scheduler
#     config.scheduler = optim.lr_scheduler.ExponentialLR(
#         config.optimizer,
#         gamma=config.scheduler_gamma,
#     )
#
#     print(config)
#
#     # # create dataset and dataloader
#     # train_set, val_set, benchmark_set = get_datasets(config)
#     # config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
#     #                                                           batch_size=config.batch_size,
#     #                                                           shuffle=True,
#     #                                                           num_workers=config.num_workers,
#     #                                                           )
#     # config.val_loader, _ = get_dataloader(dataset=val_set,
#     #                                       batch_size=config.batch_size,
#     #                                       shuffle=False,
#     #                                       num_workers=1,
#     #                                       neighborhood_limits=neighborhood_limits
#     #                                       )
#     # config.test_loader, _ = get_dataloader(dataset=benchmark_set,
#     #                                        batch_size=config.batch_size,
#     #                                        shuffle=False,
#     #                                        num_workers=1,
#     #                                        neighborhood_limits=neighborhood_limits)
#     #
#     # # create evaluation metrics
#     # config.desc_loss = MetricLoss(config)
#     # trainer = get_trainer(config)
#     # if (config.mode == 'train'):
#     #     trainer.train()
#     # elif (config.mode == 'val'):
#     #     trainer.eval()
#     # else:
#     #     trainer.test()


import pickle
import torch
import numpy as np
import open3d as o3d



# 'data/indoor/train/rgbd-scenes-v2-scene_01/cloud_bin_1.pth'
if __name__=='__main__':
    # with open('/d2/code/gitRes/Expts/OverlapPredator-main/configs/indoor/train_info.pkl','rb') as f:
    #     res_pk=pickle.load(f)
    # print(type(res_pk))
    # print(res_pk)

    # SAME
    # src=torch.load('data/indoor/train/7-scenes-chess/cloud_bin_0.pth')
    # print(type(src))
    # x=torch.load('testAPI/data_test/cloud_bin_0.pth')
    # print(type(x))
    # y = torch.load('testAPI/data_test/cloud_bin_0.pkl')
    # print(type(y))

    # x=np.random.rand(1)
    # y=np.random.rand(1)[0]
    # print("x={}".format(x))
    # print("y={}".format(y))

    # pt1 = np.random.rand(300, 3)
    # pt2 = np.random.rand(800, 3)
    # visualize_pcd(pt1,pt2)

    # trans_mat=np.eye(6)
    # trans1=np.random.rand(3,1)
    # trans2=np.random.rand(3)
    # print(trans_mat)
    # print(trans1)
    # trans_mat[3,:3]=trans1.flatten()
    # trans_mat[2:5,3]=trans2.flatten()

    # v=torch.ones(1)
    # print(v)

    # lst1=["xxx",2,345,"d542saf"]
    # lst_tmp=["coco",78]
    # lst1+=lst_tmp
    # print(lst1)

    # t1=torch.randint(low=2,high=33,size=(10,3),dtype=torch.int64,requires_grad=False)
    # print('t1: ',t1)
    # maxnumt1=t1.max()
    # print('maxnumt1: ', maxnumt1)
    # i1=torch.randint(low=2,high=10,size=(9,5),dtype=torch.int64,requires_grad=False)
    # print('i1: ',i1)
    # r1=t1[i1,:]
    # print('r1: ',r1)

    # # torch.matmul
    # data1=torch.randint(2,100,size=(1,275,314),requires_grad=False)
    # data2=torch.randint(411,555,size=(1,314,256),requires_grad=False)
    # data_res=torch.matmul(data1,data2)
    # print(data_res.shape)


    # crt=np.random.randint(low=2,high=10,size=(8,3))
    # print('crt: {}'.format(crt))
    # crt2=list(set(crt[:, 0].tolist()))
    # print('crt2: {}'.format(crt2))

    a=torch.randint(3,40,(5,7),dtype=torch.int32)
    print('a: ',a)
    a1=torch.randint(30,40,(1,7),dtype=torch.int32)
    print('a1: ', a1)
    a2=torch.randint(10,20,(1,7),dtype=torch.int32)
    print('a2: ', a2)

    b=torch.cat([a1,a2,a],dim=0)
    print('b: ', b)
    pass




