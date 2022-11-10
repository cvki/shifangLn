# from copyreg import pickle
import random
from cProfile import label
from multiprocessing import set_forkserver_preload
import h5py
import numpy as np
import open3d as o3d
import os
import torch
from utils.file_utils import load_pickle

from torch.utils.data import Dataset
import sys
import pickle
# from testAPI import randomDir
from utils.dataGenerator_multiprocess import randomItemGenerator, randomDir
from utils.plot_utils import visualize_pc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform, random_crop


'Train: file_path="/d2/code/datas/MyModelNet40/train", Test:file_path="/d2/code/datas/MyModelNet40/test'
class MyModelNet40(Dataset):
    def __init__(self,root,train_mode,normal=False, mode=None,point_num=4096, size=None):
        if train_mode=='train':
            self.file_dir=os.path.join(root, 'train')
        elif train_mode=='valid':
            self.file_dir=os.path.join(root, 'valid')
        else:
            self.file_dir=os.path.join(root,'test')
        self.train_mode=train_mode
        self.normal=normal
        self.name_datas=os.listdir(self.file_dir)
        self.point_num = point_num
        self.size = size
                     
    def __len__(self):      
        return len(self.name_datas) if self.size is None else self.size

    def __getitem__(self, index):
        if self.train_mode=='train':
            index = random.randint(0, len(self.name_datas)-1)
        datas=load_pickle(os.path.join(self.file_dir, self.name_datas[index]))
        pts1,pts2=datas['pts1'],datas['pts2']
        # pts1,pts2,quat_vec=datas['pts2'],datas['pts1'],datas['quat_vec']
        R, T = datas['trans_mat'], datas['trans_vec']
        # visualize_pc(pts1)
        # visualize_pc(pts2)
        pts1 = pts1[np.random.randint(0, pts1.shape[0], (self.point_num))]
        pts2 = pts2[np.random.randint(0, pts2.shape[0], (self.point_num))]
        return pts1, pts2, R, T.astype(np.float32)
  
    
    
    
# if __name__=='__main__':
#     file_dir='/d2/code/datas/MyModelNet40/train/'
#     mmdn=MyModelNet40(file_dir,train=True)

    
# print(len(mmdn))
# print(mmdn.quat_vec)
# print(mmdn.trans_mat)
# print(mmdn.trans_vec)
# print(mmdn.pts1)
# print(mmdn.pts2)


# class ModelNet40(Dataset):
#     def __init__(self, root, npts, train=True, normal=False, mode='clean'):
#         super(ModelNet40, self).__init__()
#         self.npts = npts        
#         self.train = train      # bool
#         self.normal = normal    # bool
#         self.mode = mode        # string to select mode
#         files = [os.path.join(root, 'ply_data_train{}.h5'.format(i))
#                  for i in range(5)]
#         if not train:
#             files = [os.path.join(root, 'ply_data_test{}.h5'.format(i))
#                      for i in range(2)]
#         self.data, self.labels = self.decode_h5(files)

#     def decode_h5(self, files):
#         points, normal, label = [], [], []
#         for file in files:
#             f = h5py.File(file, 'r')
#             cur_points = f['data'][:].astype(np.float32)
#             cur_normal = f['normal'][:].astype(np.float32)
#             cur_label = f['label'][:].astype(np.float32)
#             points.append(cur_points)
#             normal.append(cur_normal)
#             label.append(cur_label)
#         points = np.concatenate(points, axis=0)
#         normal = np.concatenate(normal, axis=0)
#         data = np.concatenate([points, normal], axis=-1).astype(np.float32)
#         label = np.concatenate(label, axis=0)
#         return data, label
       
#     def compose(self, mode, item):
#         ref_cloud = self.data[item, ...]
#         R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
#         if mode == 'clean':
#             ref_cloud = random_select_points(ref_cloud, m=self.npts)
#             src_cloud_points = transform(ref_cloud[:, :3], R, t)
#             src_cloud_normal = transform(ref_cloud[:, 3:], R)
#             src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
#                                        axis=-1)
#             return src_cloud, ref_cloud, R, t
#         elif mode == 'partial':
#             source_cloud = random_select_points(ref_cloud, m=self.npts)
#             ref_cloud = random_select_points(ref_cloud, m=self.npts)
#             src_cloud_points = transform(source_cloud[:, :3], R, t)
#             src_cloud_normal = transform(source_cloud[:, 3:], R)
#             src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
#                                        axis=-1)
#             src_cloud = random_crop(src_cloud, p_keep=0.7)
#             return src_cloud, ref_cloud, R, t
#         elif mode == 'noise':
#             source_cloud = random_select_points(ref_cloud, m=self.npts)
#             ref_cloud = random_select_points(ref_cloud, m=self.npts)
#             src_cloud_points = transform(source_cloud[:, :3], R, t)
#             src_cloud_normal = transform(source_cloud[:, 3:], R)
#             src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
#                                        axis=-1)
#             return src_cloud, ref_cloud, R, t
#         else:
#             raise NotImplementedError

#     def __getitem__(self, item):
#         src_cloud, ref_cloud, R, t = self.compose(mode=self.mode, item=item)
#         if self.train or self.mode == 'noise' or self.mode == 'partial':
#             ref_cloud[:, :3] = jitter_point_cloud(ref_cloud[:, :3])
#             src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])
#         if not self.normal:
#             ref_cloud, src_cloud = ref_cloud[:, :3], src_cloud[:, :3]
#         return ref_cloud, src_cloud, R, t

#     def __len__(self):
#         return len(self.data)