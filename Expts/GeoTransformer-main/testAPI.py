# import h5py
#
# def lookpts(path):
#     hf=h5py.File(path,'r')
#     for item in hf:
#         for v in hf[item]:
#             print(v.shape)
#
# modelnet40, [data]: 2048个点云,每个点云中,2048个点
# lookpts('/d2/code/gitRes/Expts/GeoTransformer-main/data/ModelNet/modelnet40_ply_hdf5_2048/ply_data_train0.h5')

import numpy as np

# a=np.array([i for i in range(21)]).reshape(3,7)
# print(a)
# b=[np.sum(a<a.shape[1],axis=1)]
# print(b)

import open3d as o3d
import torch


def visualizepc1(pts,str='pcd'):   # np--pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts)
    pcd1.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([pcd1],window_name=str)

def visualizepc2(pts1,pts2, str='pcd1,pcd2'):  # np--pts
    if isinstance(pts1,torch.Tensor):
        pts1=pts1.numpy()
    if isinstance(pts2,torch.Tensor):
        pts2=pts2.numpy()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1,0,0])   # red
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0,1,0])   # green
    o3d.visualization.draw_geometries([pcd1,pcd2], window_name=str)

# def visualizepc3(*pts, str='pcd1-3'):  # np--pts
def visualizepc3(pts1,pts2,pts3, str='pcd1-3'):  # np--pts
    # pts1,pts2,pts3=pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1,0,0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0,1,0])
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pts3)
    pcd3.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], window_name=str)


'''from ._ckdtree import cKDTree, cKDTreeNode  
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' ' \
'not found (required by /home/sfhc/anaconda3/envs/GeoTfm/lib/python3.8/site-packages/scipy/spatial/_ckdtree.cpython-38-x86_64-linux-gnu.so)  '''
if __name__=='__main__':
    pc1=np.random.randn(11,3)
    pc2=np.random.randn(25,3)
    pc3=np.random.randn(17,3)
    visualizepc1(pc1,'TestAPI')
    visualizepc2(pc1,pc2,'TestAPI')
    visualizepc3(pc1,pc2,pc3,'TestAPI')






