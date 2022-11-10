
import os
import numpy as np
import torch
from utils.utils import readTXT2Numpy,PointcloudRandomCropTwice,PointcloudRotate,PointcloudTranslate,\
    mat2quat,load_pickle,save_pickle2, remove_and_mkdir,jitter_point_cloud,run_imap_multiprocessing
# from utils.utils import npy2pcd
# import open3d as o3d

class DataGenerator:
    def __init__(self, num_datas, mode, source_path="/d2/code/datas/modelnet40_normal_resampled/",
                 result_path='/d2/code/datas/vs_modelnet40/',
                 txt_describe="modelnet40_shape_names.txt"):
        self.num_datas = num_datas
        self.mode = mode
        self.source_path = source_path
        self.result_path = result_path
        self.txt_describe = txt_describe

    def randomDir(self):
        from_path = os.path.join(self.source_path, self.txt_describe)
        labels_dir = []
        with open(from_path, 'r') as f:
            for line in f:
                labels_dir.append(line.strip())
        idx_tmp = np.random.uniform(0, len(labels_dir))
        self.label = labels_dir[int(idx_tmp)]
        # print(labels_dir)
        return self.label

    def randomFile(self):
        label = self.randomDir()
        filepath = os.path.join(self.source_path, label)
        txt_files = os.listdir(filepath)
        file_idx_tmp = np.random.randint(0, len(txt_files))
        return label, os.path.join(filepath, txt_files[file_idx_tmp])

    def randomItemGenerator(self, source_file):
        points = readTXT2Numpy(source_file, diminator=',')  # (N, 6)
        points = points[:, :3]
        randomCropTwice = PointcloudRandomCropTwice()
        pts1, pts2, idx1, idx2 = randomCropTwice(points)
        if pts1 is None:
            return None, None, None, None, None, None, None

        # print(np.shape(pts1),'\t',np.shape(pts2),'\t',np.shape(idx1),'\t',np.shape(idx2))

        rotate_trans = PointcloudRotate()
        pts1_trans, trans_metrix = rotate_trans(torch.from_numpy(pts1))
        quat_vec = mat2quat(trans_metrix)

        parral_trans = PointcloudTranslate()
        pts1_trans, trans_vector = parral_trans(pts1_trans)

        # visualize
        # pcd1=npy2pcd(pts1_trans,0)   # red
        # pcd2=npy2pcd(pts2,1)   # green
        # o3d.visualization.draw_geometries([pcd1,pcd2],window_name=self.label+':\t'+str(pts1_trans.size/3)+', '+str(pts2.size/3))

        # add Gauss noise
        pts1_trans = jitter_point_cloud(pts1_trans)
        pts2 = jitter_point_cloud(pts2)

        # visualize
        # pcd1 = npy2pcd(pts1_trans, 0)  # red
        # pcd2 = npy2pcd(pts2, 1)  # green
        # o3d.visualization.draw_geometries([pcd1, pcd2],
        #                                   window_name=self.label + ':\t' + str(pts1_trans.size / 3) + ', ' + str(pts2.size / 3))

        # gauss_noise=np.random.normal(0,0.45,(np.shape(pts1_trans)))
        # pts1_trans+=gauss_noise
        quat_vec = np.append(quat_vec, trans_vector).reshape(1, 7)

        return pts1_trans, pts2, idx1, idx2, trans_metrix, trans_vector, quat_vec

    def process(self, data_sign):
        import time
        # np.random.seed(data_sign * time.time())  # seed
        label, source_file = self.randomFile()
        # try:
        pts1_trans, pts2, index1, index2, trans_metrix, trans_vector, quat = self.randomItemGenerator(source_file)
        # except:
        #     # print(source_file)
        #     pts1_trans = None
        if pts1_trans is None:
            return
        path_dir = self.result_path + '/' + self.mode
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        pkl_file_pts = self.result_path + '/' + self.mode + '/pts_' + str(data_sign) + '.pkl'
        save_pickle2(pkl_file_pts,
                     {'pts1': pts1_trans, 'pts2': pts2, 'idx1': index1, 'idx2': index2, 'trans_mat': trans_metrix,
                      'trans_vec': trans_vector,
                      'quat_vec': quat, 'label': label})

source_path="/d2/code/datas/modelnet40_normal_resampled"
result_path='/d2/code/datas/vs_modelnet40'
txt_describe="modelnet40_shape_names.txt"
remove_and_mkdir(result_path)

data_train = DataGenerator(num_datas=30000, mode='train')
data_valid=DataGenerator(num_datas=7000,mode='valid')
data_test=DataGenerator(num_datas=7000,mode='test')

run_imap_multiprocessing(data_train.process, range(30000), 8)  # train---30000
run_imap_multiprocessing(data_valid.process, range(data_valid.num_datas), 8)  # train---7000
run_imap_multiprocessing(data_test.process, range(data_test.num_datas), 8)  # train---7000


# obj = load_pickle(filename='/d2/code/datas/vs_modelnet40/train/pts_16.pkl')
# print(len(obj['idx1']))
# print(len(obj['idx2']))
# print(obj['quat_vec'])
# print(obj['trans_vec'])




