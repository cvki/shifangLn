
import pickle
import gzip
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import sys

import numpy as np
import torch
import shutil
import random
import string
import zipfile
import time
import csv
import cv2
import math
try:
    import open3d as o3d
except:
    print('open3d not found')


'''-------------------------------------------used begin---------------------------------------------------'''

def readTXT2Numpy(txt_path, diminator=','):
    with open(txt_path, 'r') as f:
        listInTXT = [[float(el) for el in line.split(diminator)] for line in f]
    return np.array(listInTXT)


def remove_and_mkdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(path, 'removed')
    os.makedirs(path)
    

def save_pickle2(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def run_imap_multiprocessing(func, argument_list, num_processes):
    from multiprocessing import Pool
    from tqdm import tqdm
    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm


def quat2mat(quat):
    w, x, y, z = quat
    R = np.zeros((3, 3), dtype=np.float32)
    R[0][0] = 1 - 2*y*y - 2*z*z
    R[0][1] = 2*x*y - 2*z*w
    R[0][2] = 2*x*z + 2*y*w
    R[1][0] = 2*x*y + 2*z*w
    R[1][1] = 1 - 2*x*x - 2*z*z
    R[1][2] = 2*y*z - 2*x*w
    R[2][0] = 2*x*z - 2*y*w
    R[2][1] = 2*y*z + 2*x*w
    R[2][2] = 1 - 2*x*x - 2*y*y
    return R


def batch_quat2mat(batch_quat):
    '''
    :param batch_quat: shape=(B, 4)
    :return:
    '''
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R


def mat2quat(mat):
    w = math.sqrt(mat[0, 0] + mat[1, 1] + mat[2, 2] + 1) / 2
    x = (mat[2, 1] - mat[1, 2]) / (4 * w)
    y = (mat[0, 2] - mat[2, 0]) / (4 * w)
    z = (mat[1, 0] - mat[0, 1]) / (4 * w)
    return w, x, y, z


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    if isinstance(pc,torch.Tensor):
        pc=pc.numpy()
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
    # print(np.shape(jittered_data))
    jittered_data += pc
    return jittered_data


def to_color_map(diff, type=cv2.COLORMAP_JET):
    diff = diff / diff.max() * 255.0
    diff = diff.astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def is_image_file(filename, suffix=None):
    if suffix is not None:
        IMG_EXTENSIONS = [suffix]
    else:
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npz'
        ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, suffix=None, max=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, suffix):
                path = os.path.join(root, fname)
                images.append(path)
    if max is not None:
        return images[:max]
    else:
        return images


def visualize_pc(pts, colors=None, size=0.3, window_name='Open3D'):
    # pts (n, 3) numpy
    # colors (n, 3) in RGB
    print(pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        if colors.max() > 1:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)], window_name=window_name)
    # exit(0)
    
    
class PointcloudRotate(object):
    def __call__(self, points):
        angles = np.random.uniform(size=3) * 2 * np.pi
        Rx = self.angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = self.angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = self.angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        # points[:, 0:3] = torch.matmul(points[:, 0:3], rotation_matrix.t())
        points = torch.matmul(points, rotation_matrix.t().to(torch.float64))
        return points,rotation_matrix
    
    def angle_axis(self,angle, axis):
        # type: (float, np.ndarray) -> float
        r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
        Parameters
        ----------
        angle : float
            Angle to rotate by
        axis: np.ndarray
            Axis to rotate about
        Returns
        -------
        torch.Tensor
            3x3 rotation matrix
        """
        u = axis / np.linalg.norm(axis)
        cosval, sinval = np.cos(angle), np.sin(angle)

        # yapf: disable
        cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])
        R = torch.from_numpy(
            cosval * np.eye(3)
            + sinval * cross_prod_mat
            + (1.0 - cosval) * np.outer(u, u)
        )
        # yapf: enable
        return R.float()

    
class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        if isinstance(points,torch.Tensor):
            points = points.numpy()
        coord_min = np.min(points[:, :3], axis=0)
        coord_max = np.max(points[:, :3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        points[:, 0:3] += translation
        return points,translation
    
    
class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()
    
    
class PointcloudRandomCropTwice(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33,max_try_num=30,min_num_points=6500,max_num_points=7500):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points

    def __call__(self, points):
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        count_pts=0
        pts,idx=[],[] 
        while count_pts<2:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]
            
            self.max_try_num-=1
            if self.max_try_num <1:
                # raise NotImplementedError
                pts = []
                break                
            # print('new_points: ',np.shape(new_points))
            if(np.shape(new_points)[0]>self.min_num_points and np.shape(new_points)[0]<self.max_num_points):
                pts.append(new_points.tolist())
                idx.append(np.argwhere(new_indices))
                count_pts+=1
        self.max_try_num=30
        if len(pts) < 1:
            return None, None, None, None
        return np.array(pts[0]),np.array(pts[1]),np.array(idx[0]),np.array(idx[1])

    
    
    
'''-------------------------------------------used end---------------------------------------------------'''



























# """
# Author: Zhenbo Xu
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# """
# def readTXT(txt_path):
#     with open(txt_path, 'r') as f:
#         listInTXT = [line.strip() for line in f]
#     return listInTXT


# def get_csv_content(csv_path, delimiter=',', remove_ind=True):
#     with open(csv_path, newline='') as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=',')
#         content = []
#         for row in enumerate(spamreader):
#             content.append(row if not remove_ind else row[1])
#         return content


# def write_rows_to_csv(csv_file_path, rows):
#     assert isinstance(rows, list)
#     # open the file in the write mode
#     f = open(csv_file_path, 'w')
#     # create the csv writer
#     writer = csv.writer(f)
#     # write a row to the csv file
#     for row in rows:
#         writer.writerow(row)
#     # close the file
#     f.close()


# def get_random_string(length):
#     # choose from all lowercase letter
#     letters = string.ascii_lowercase
#     result_str = ''.join(random.choice(letters) for i in range(length))
#     # print("Random string of length", length, "is:", result_str)
#     return result_str


# def listdir_nohidden(path):
#     for f in os.listdir(path):
#         if not f.startswith('.'):
#             yield f


# # def read_profile(model=None, input=None):
# #     from thop import profile, clever_format
# #     if model == None:
# #         from torchvision.models import resnet50
# #         model = resnet50()
# #     if input == None:
# #         input = torch.randn(1, 3, 224, 224)
# #     macs, params = profile(model, inputs=(input,))
# #     macs, params = clever_format([macs, params], "%.3f")
# #     return macs, params


# def remove_key_word(previous_dict, keywords):
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in previous_dict.items():
#         if_exist_keyword = [1 if el in k else 0 for el in keywords]
#         if sum(if_exist_keyword) == 0:
#             new_state_dict[k] = v
#     return new_state_dict


# def load_weights_from_data_parallel(model_path, net):
#     if not os.path.isfile(model_path):
#         print('%s not found' % model_path)
#         exit(0)
#     else:
#         print('Load from %s' % model_path)
#     previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in previous_dict.items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     net.load_state_dict(new_state_dict, strict=True)
#     return net


# def remove_module_in_dict(loaded_dict):
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in loaded_dict.items():
#         # name = k[7:]  # remove `module.`
#         try:
#             if k.split('.', 1)[0] == 'model' or k.split('.', 1)[0] == 'backbone' or k.split('.', 1)[0] == 'module':
#                 name = k.split('.', 1)[1]
#                 new_state_dict[name] = v
#         except:
#             print(k, 'convert fail')
#     return new_state_dict


# def load_weights(model_path, net, strict=True):
#     if not os.path.isfile(model_path):
#         print('%s not found' % model_path)
#         exit(0)
#     else:
#         print('Load from %s' % model_path)
#     previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
#     net.load_state_dict(previous_dict, strict=strict)
#     return net


# def mkdir_if_no(path):
#     if not os.path.isdir(path):
#         os.makedirs(path)


# def save_zipped_pickle(obj, filename, protocol=-1):
#     with gzip.open(filename, 'wb') as f:
#         pickle.dump(obj, f, protocol)
#     return


# def load_zipped_pickle(filename):
#     with gzip.open(filename, 'rb') as f:
#         loaded_object = pickle.load(f)
#     return loaded_object


# def save_pickle(filename, obj):
#     with open(filename, 'wb') as f:
#         pickle.dump(obj, f)


# def load_json(filename):
#     return json.load(open(filename, 'r'))


# def save_json(filename, res):
#     json.dump(res, open(filename, 'w'))


# def save_json_with_np(filename, res):
#     import numpy as np
#     class NpEncoder(json.JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, np.integer):
#                 return int(obj)
#             elif isinstance(obj, np.floating):
#                 return float(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             else:
#                 return super(NpEncoder, self).default(obj)

#     json.dump(res, open(filename, 'w'), cls=NpEncoder)



# def make_dataset_prefix(dir, prefix):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir

#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if fname.startswith(prefix):
#                 path = os.path.join(root, fname)
#                 images.append(path)
#     return images


# def describe_element(name, df):
#     """ Takes the columns of the dataframe and builds a ply-like description
#     Parameters
#     ----------
#     name: str
#     df: pandas DataFrame
#     Returns
#     -------
#     element: list[str]
#     """
#     property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
#     element = ['element ' + name + ' ' + str(len(df))]

#     if name == 'face':
#         element.append("property list uchar int vertex_indices")

#     else:
#         for i in range(len(df.columns)):
#             # get first letter of dtype to infer format
#             f = property_formats[str(df.dtypes[i])[0]]
#             element.append('property ' + f + ' ' + df.columns.values[i])

#     return element


# def write_ply(save_path,points,text=True):
#     """
#     save_path : path to save: '/yy/XX.ply'
#     pt: point_cloud: size (N,3)
#     """
#     from plyfile import PlyData, PlyElement
#     points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
#     vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
#     el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
#     PlyData([el], text=text).write(save_path)


# def write_to_file(dst_path, upload_content):
#     with open(dst_path, "w") as f:
#         for line in upload_content:
#             print(line, file=f)


# class obj:
#     # constructor
#     def __init__(self, dict1):
#         self.__dict__.update(dict1)


# def dict2obj(dict1):
#     # using json.loads method and passing json.dumps
#     # method and custom object hook as arguments
#     return json.loads(json.dumps(dict1), object_hook=obj)


# def getSubDirs(path, shuffle=False):
#     res = [f.path for f in os.scandir(path) if f.is_dir()]
#     if shuffle:
#         random.shuffle(res)
#     return res


# def getSubSubDirs(path):
#     subdirs = getSubDirs(path)
#     subsubdirs = []
#     for subdir in subdirs:
#         subsubdirs += getSubDirs(subdir)
#     return subsubdirs


# def getSubDirNames(srcRoot):
#     return [el for el in os.listdir(srcRoot) if os.path.isdir(os.path.join(srcRoot, el))]


# def unzip_with_timestamp(inFile, outDirectory):
#     # outDirectory = 'C:\\TEMP\\'
#     # inFile = 'test.zip'
#     fh = open(os.path.join(outDirectory, inFile), 'rb')
#     z = zipfile.ZipFile(fh)

#     for f in z.infolist():
#         name, date_time = f.filename, f.date_time
#         name = os.path.join(outDirectory, name)
#         with open(name, 'wb') as outFile:
#             outFile.write(z.open(f).read())
#         date_time = time.mktime(date_time + (0, 0, -1))
#         os.utime(name, (date_time, date_time))

# def write_points_colors_to_file(save_path, points, colors):
#     # pc = o3d.io.read_point_cloud(input_file)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     if colors is not None:
#         if colors.max() > 1:
#             colors = colors.astype(np.float32) / 255.0
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_point_cloud(save_path, pcd)
    

# def visualize_pcd(pcd, size=0.3):
#     print(pcd.points)
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#     o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)])
#     # exit(0)
    
    
# # import copy
# # import cv2
# # import numpy as np
# # import torch
# # import time
# # from random import random
# # try:
# #     import open3d as o3d
# # except:
# #     print('open3d load fail')




# def fps(points, num):
#     cids = []
#     cid = np.random.choice(points.shape[0])
#     cids.append(cid)
#     id_flag = np.zeros(points.shape[0])
#     id_flag[cid] = 1

#     dist = torch.zeros(points.shape[0]) + 1e4
#     dist = dist.type_as(points)
#     while np.sum(id_flag) < num:
#         dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
#         dist = torch.where(dist < dist_c, dist, dist_c)
#         dist[id_flag == 1] = 1e4
#         new_cid = torch.argmin(dist)
#         id_flag[new_cid] = 1
#         cids.append(new_cid)
#     cids = torch.Tensor(cids)
#     return cids


# class PointcloudScale(object):
#     def __init__(self, lo=0.8, hi=1.25, p=1):
#         self.lo, self.hi = lo, hi
#         self.p = p

#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#         scaler = np.random.uniform(self.lo, self.hi)
#         points[:, 0:3] *= scaler
#         return points


# class PointcloudRotatePerturbation(object):
#     def __init__(self, angle_sigma=0.06, angle_clip=0.18, p=1):
#         self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
#         self.p = p

#     def _get_angles(self):
#         angles = np.clip(
#             self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
#         )

#         return angles

#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#         angles = self._get_angles()
#         Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
#         Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
#         Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

#         rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

#         normals = points.size(1) > 3
#         if not normals:
#             return torch.matmul(points, rotation_matrix.t())
#         else:
#             pc_xyz = points[:, 0:3]
#             pc_normals = points[:, 3:]
#             points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
#             points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

#             return points


# class PointcloudJitter(object):
#     def __init__(self, std=0.01, clip=0.05, p=1):
#         self.std, self.clip = std, clip
#         self.p = p

#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#         jittered_data = (
#             points.new(points.size(0), 3)
#                 .normal_(mean=0.0, std=self.std)
#                 .clamp_(-self.clip, self.clip)
#         )
#         points[:, 0:3] += jittered_data
#         return points




# class PointcloudToTensor(object):
#     def __call__(self, points):
#         return torch.from_numpy(points).float()


# class PointcloudRandomInputDropout(object):
#     def __init__(self, max_dropout_ratio=0.875, p=1):
#         assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
#         self.max_dropout_ratio = max_dropout_ratio
#         self.p = p

#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#         pc = points.numpy()

#         dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
#         drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
#         if len(drop_idx) > 0:
#             pc[drop_idx] = pc[0]  # set to the first point

#         return torch.from_numpy(pc).float()


# class PointcloudSample(object):
#     def __init__(self, num_pt=4096):
#         self.num_points = num_pt

#     def __call__(self, points):
#         pc = points.numpy()
#         # pt_idxs = np.arange(0, self.num_points)
#         pt_idxs = np.arange(0, points.shape[0])
#         np.random.shuffle(pt_idxs)
#         pc = pc[pt_idxs[0:self.num_points], :]
#         return torch.from_numpy(pc).float()





# class PointcloudRemoveInvalid(object):
#     def __init__(self, invalid_value=0):
#         self.invalid_value = invalid_value

#     def __call__(self, points):
#         pc = points.numpy()
#         valid = np.sum(pc, axis=1) != self.invalid_value
#         pc = pc[valid, :]
#         return torch.from_numpy(pc).float()


# def random_rgb_generator(float=True, avoid_black=False):
#     if avoid_black:
#         res = np.array([int((random()*0.9+0.1)*255)/255, int((random()*0.9+0.1)*255)/255, int((random()*0.9+0.1)*255)/255])
#     else:
#         res = np.array([int(random()*255)/255, int(random()*255)/255, int(random()*255)/255])
#     if not float:
#         return (res*255).astype(np.uint8)
#     return res


# class PointcloudRandomColor(object):
#     def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, try_num=10):
#         self.x_min = x_min
#         self.x_max = x_max

#         self.ar_min = ar_min
#         self.ar_max = ar_max

#         self.p = p

#         self.max_try_num = try_num

#     def __call__(self, points):
#         if isinstance(points, torch.Tensor):
#             points = points.numpy()

#         try_num = 0
#         colors = np.ones(points.shape) * random_rgb_generator(avoid_black=True)[np.newaxis]
#         coord_min = np.min(points[:, :3], axis=0)
#         coord_max = np.max(points[:, :3], axis=0)
#         coord_diff = coord_max - coord_min
#         while try_num < self.max_try_num:
#             # resampling later, so only consider crop here
#             new_coord_range = np.zeros(3)
#             new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
#             ar = np.random.uniform(self.ar_min, self.ar_max)
#             # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
#             # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
#             new_coord_range[1] = new_coord_range[0] * ar
#             new_coord_range[2] = new_coord_range[0] / ar

#             new_coord_min = np.random.uniform(0, 1 - new_coord_range)
#             new_coord_max = new_coord_min + new_coord_range

#             new_coord_min = coord_min + coord_diff * new_coord_min
#             new_coord_max = coord_min + coord_diff * new_coord_max

#             new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
#             random_color = random_rgb_generator(avoid_black=True)[np.newaxis]
#             # if random() < 0.5:
#             new_indices = np.sum(new_indices, axis=1) == 3
#             colors[new_indices] = random_color
#             # else:
#             #     new_indices = np.sum(new_indices, axis=1) < 3
#             #     colors[new_indices] = random_color[np.newaxis]
#             try_num += 1
#         # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
#         # return torch.from_numpy(new_points).float()
#         return points, colors


# class PointcloudRandomCutout(object):
#     def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
#         self.ratio_min = ratio_min
#         self.ratio_max = ratio_max
#         self.p = p
#         self.min_num_points = min_num_points
#         self.max_try_num = max_try_num

#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#         points = points.numpy()
#         try_num = 0
#         valid = False
#         while not valid:
#             coord_min = np.min(points[:, :3], axis=0)
#             coord_max = np.max(points[:, :3], axis=0)
#             coord_diff = coord_max - coord_min

#             cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
#             new_coord_min = np.random.uniform(0, 1 - cut_ratio)
#             new_coord_max = new_coord_min + cut_ratio

#             new_coord_min = coord_min + new_coord_min * coord_diff
#             new_coord_max = coord_min + new_coord_max * coord_diff

#             cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
#             cut_indices = np.sum(cut_indices, axis=1) == 3

#             # print(np.sum(cut_indices))
#             # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
#             # other_indices = np.sum(other_indices, axis=1) == 3
#             try_num += 1

#             if try_num > self.max_try_num:
#                 return torch.from_numpy(points).float()

#             # cut the points, sampling later

#             if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
#                 # print (np.sum(cut_indices))
#                 points = points[cut_indices == False]
#                 valid = True

#         # if np.sum(other_indices) > 0:
#         #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
#         #     points[cut_indices] = points[comp_indices]
#         return torch.from_numpy(points).float()


# class PointcloudUpSampling(object):
#     def __init__(self, max_num_points, radius=0.1, nsample=5, centroid="random"):
#         self.max_num_points = max_num_points
#         # self.radius = radius
#         self.centroid = centroid
#         self.nsample = nsample

#     def __call__(self, points):
#         t0 = time.time()

#         p_num = points.shape[0]
#         if p_num > self.max_num_points:
#             return points

#         c_num = self.max_num_points - p_num

#         if self.centroid == "random":
#             cids = np.random.choice(np.arange(p_num), c_num)
#         else:
#             assert self.centroid == "fps"
#             fps_num = c_num / self.nsample
#             fps_ids = fps(points, fps_num)
#             cids = np.random.choice(fps_ids, c_num)

#         xyzs = points[:, :3]
#         loc_matmul = torch.matmul(xyzs, xyzs.t())
#         loc_norm = xyzs * xyzs
#         r = torch.sum(loc_norm, -1, keepdim=True)

#         r_t = r.t()  # 转置
#         dist = r - 2 * loc_matmul + r_t
#         # adj_matrix = torch.sqrt(dist + 1e-6)

#         dist = dist[cids]
#         # adj_sort = torch.argsort(adj_matrix, 1)
#         adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]

#         uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
#         median = np.median(uniform, axis=1, keepdims=True)
#         # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
#         choice = adj_topk[uniform > median]  # (c_num, n_samples)

#         choice = choice.reshape(-1, self.nsample)

#         sample_points = points[choice]  # (c_num, n_samples, 3)

#         new_points = torch.mean(sample_points, dim=1)
#         new_points = torch.cat([points, new_points], 0)

#         return new_points


# class PointcloudBatchUpSampling(object):
#     def __init__(self, max_num_points, radius=0.1, nsample=5, centroid="random"):
#         self.max_num_points = max_num_points
#         # self.radius = radius
#         self.centroid = centroid
#         self.nsample = nsample

#     def __call__(self, pointsList):
#         # t0 = time.time()
#         newList = []
#         for points in pointsList:
#             p_num = points.shape[0]
#             if p_num > self.max_num_points:
#                 return points

#             c_num = self.max_num_points - p_num

#             if self.centroid == "random":
#                 # cids = np.random.choice(np.arange(p_num), c_num)
#                 cids = torch.randint(points.shape[0], (c_num,)).to(points.device)
#             else:
#                 assert self.centroid == "fps"
#                 fps_num = c_num / self.nsample
#                 fps_ids = fps(points, fps_num)
#                 # cids = np.random.choice(fps_ids, c_num)
#                 cids = torch.randint(fps_ids.shape[0], (c_num,)).to(points.device)

#             xyzs = points[:, :3]
#             loc_matmul = torch.matmul(xyzs, xyzs.t())
#             loc_norm = xyzs * xyzs
#             r = torch.sum(loc_norm, -1, keepdim=True)

#             r_t = r.t()  # 转置
#             dist = r - 2 * loc_matmul + r_t
#             # adj_matrix = torch.sqrt(dist + 1e-6)

#             dist = dist[cids]
#             # adj_sort = torch.argsort(adj_matrix, 1)
#             adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]

#             # uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
#             uniform_torch = torch.FloatTensor(cids.shape[0], self.nsample * 2).uniform_(0, 1).to(points.device)
#             # median = np.median(uniform, axis=1, keepdims=True)
#             median_torch = torch.median(uniform_torch, dim=1, keepdim=True)[0].to(points.device)
#             choice = adj_topk[uniform_torch > median_torch]  # (c_num, n_samples)

#             choice = choice.reshape(-1, self.nsample)

#             sample_points = points[choice]  # (c_num, n_samples, 3)

#             new_points = torch.mean(sample_points, dim=1)
#             new_points = torch.cat([points, new_points], 0)
#             newList.append(new_points)

#         return torch.stack(newList)


# def points_sampler(points, num):
#     pt_idxs = np.arange(0, points.shape[0])
#     np.random.shuffle(pt_idxs)
#     points = points[pt_idxs[0:num], :]
#     return points


# class PointcloudScaleAndTranslate(object):
#     def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
#         self.scale_low = scale_low
#         self.scale_high = scale_high
#         self.translate_range = translate_range

#     def __call__(self, pc, device):
#         bsize = pc.size()[0]
#         dim = pc.size()[-1]

#         for i in range(bsize):
#             xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[dim])
#             xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[dim])

#             pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(device)) + torch.from_numpy(
#                 xyz2).float().to(device)

#         return pc


# def roty(t):
#     ''' Rotation about the y-axis. '''
#     c = np.cos(t)
#     s = np.sin(t)
#     return np.array([[c,  0,  s],
#                      [0,  1,  0],
#                      [-s, 0,  c]])


# def project_to_image(pts_3d, P):
#     ''' Project 3d points to image plane.

#     Usage: pts_2d = projectToImage(pts_3d, P)
#       input: pts_3d: nx3 matrix
#              P:      3x4 projection matrix
#       output: pts_2d: nx2 matrix

#       P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
#       => normalize projected_pts_2d(2xn)

#       <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
#           => normalize projected_pts_2d(nx2)
#     '''
#     n = pts_3d.size()[0]
#     pts_3d_extend = torch.cat([pts_3d, torch.ones((n,1)).float().to(pts_3d.device)], dim=1).contiguous()
#     # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
#     pts_2d = torch.matmul(pts_3d_extend, P.permute(1,0)) # nx3
#     pts_2d[:,0] /= pts_2d[:,2]
#     pts_2d[:,1] /= pts_2d[:,2]
#     return pts_2d[:,0:2]


# def rand_rotation_matrix(deflection=1.0, randnums=None):
#     """
#     Creates a random rotation matrix.

#     deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
#     rotation. Small deflection => small perturbation.
#     randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
#     """
#     # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

#     if randnums is None:
#         randnums = np.random.uniform(size=(3,))

#     theta, phi, z = randnums

#     theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
#     phi = phi * 2.0 * np.pi  # For direction of pole deflection.
#     z = z * 2.0 * deflection  # For magnitude of pole deflection.

#     # Compute a vector V used for distributing points over the sphere
#     # via the reflection I - V Transpose(V).  This formulation of V
#     # will guarantee that if x[1] and x[2] are uniformly distributed,
#     # the reflected points will be uniform on the sphere.  Note that V
#     # has length sqrt(2) to eliminate the 2 in the Householder matrix.

#     r = np.sqrt(z)
#     Vx, Vy, Vz = V = (
#         np.sin(phi) * r,
#         np.cos(phi) * r,
#         np.sqrt(2.0 - z)
#     )

#     st = np.sin(theta)
#     ct = np.cos(theta)

#     R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

#     # Construct the rotation matrix  ( V Transpose(V) - I ) R.

#     M = (np.outer(V, V) - np.eye(3)).dot(R)
#     return M


# def compute_pts_3d(pts, rand_rot_matrix, t, P):
#     ''' Written by Zhenbo Xu.
#         Takes an object and a projection matrix (P) and projects the 3d
#         bounding box into the image plane.
#         pts: input points from OBJ file, N*3(W(-0.5~0.5),H(-1~0),L(-0.5~0.5))
#         t: the GT position(l, h, w) you want to put the car in

#         Returns:
#             corners_2d: (8,2) array in left image coord.
#             corners_3d: (8,3) array in in rect camera coord.
#     '''
#     # compute rotational matrix around yaw axis
#     R = torch.from_numpy(rand_rot_matrix).float().to(pts.device)

#     # 3d bounding box dimensions:l,h,w
#     pts = torch.cat([pts[:,2:3], pts[:,1:2], pts[:,0:1]], 1).contiguous()

#     # rotate and translate 3d bounding box
#     corners_3d = torch.matmul(R, pts.permute(1,0))
#     # print corners_3d.shape
#     corners_3d[0, :] = corners_3d[0, :] + t[0]
#     corners_3d[1, :] = corners_3d[1, :] + t[1]
#     corners_3d[2, :] = corners_3d[2, :] + t[2]
#     corners_3d = corners_3d.contiguous()

#     # from plot_utils import visualize_pc, visualize_pcd
#     # visualize_pc(corners_3d.permute(1,0))

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     pcd.estimate_normals()
#     print(":: Run ball pivoting reconstruction")
#     radii = [0.005, 0.01, 0.02, 0.04]
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
#     pcd_ = mesh.sample_points_poisson_disk(1000)
#     diameter = np.linalg.norm(np.asarray(pcd_.get_max_bound()) - np.asarray(pcd_.get_min_bound()))
#     # diameter = 1.0
#     camera = [0, 0, 0]
#     radius = diameter * 100  # 1600 is better, but 800 is good
#     _, pt_map = pcd.hidden_point_removal(camera, radius)
#     pcd = pcd.select_by_index(pt_map)

#     # visualize_pcd(copy.deepcopy(pcd))

#     # project the 3d bounding box into the image plane
#     corners_2d = project_to_image(corners_3d.permute(1,0), P)
#     return corners_2d, corners_3d.permute(1,0)


# class PointcloudTransformAndProject(object):
#     def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, try_num=10, im_size=(128, 128)):
#         self.x_min = x_min
#         self.x_max = x_max

#         self.ar_min = ar_min
#         self.ar_max = ar_max
#         self.p = p
#         self.max_try_num = try_num
#         self.P = np.array([443.40496826171875,0.0,256.0,0.0,443.40496826171875,256.0,0.0,0.0,1.0]).reshape((3,3))
#         self.PT_torch = torch.from_numpy(np.concatenate([self.P, np.array([0.1,0.1,0.1])[:,np.newaxis]], axis=-1)).float() # (3*4)
#         self.w = im_size[0]
#         self.h = im_size[1]

#     def __call__(self, points, colors):
#         if isinstance(points, np.ndarray):
#             points = torch.from_numpy(points)
#         points = points.float()

#         try_num = 0
#         while try_num < self.max_try_num:
#             # ry = random() * np.pi * 2 - np.pi
#             # rand_rot_matrix = roty(ry)
#             rand_rot_matrix = rand_rotation_matrix()
#             ratio = 0.0
#             while ratio < 0.3:
#                 # cloud = o3d.geometry.PointCloud()
#                 # cloud.points = o3d.utility.Vector3dVector(points)
#                 # trimesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
#                 #
#                 # remesh_vertices, remesh_faces = subdivide_mesh(np.array(mesh.vertices), np.array(mesh.faces), 0.1)
#                 pts_2d, obj_rect_pts = compute_pts_3d(points, rand_rot_matrix, [1.0,1.0,1.0], self.PT_torch)
#                 vs, us = pts_2d[:, 0], pts_2d[:, 1]
#                 vs = (self.h * (vs - vs.min()) / (vs.max() - vs.min())).long()
#                 us = (self.w * (us - us.min()) / (us.max() - us.min())).long()

#                 # norm pts to im_size
#                 ''' t 
#                 [0,0,0]         -50w ~8w
#                 [1.0,1.0,1.0]   423~940
#                 '''
#                 img_width, img_height = self.w, self.h
#                 img_flatten_ind = torch.range(start=0, end=img_width * img_height - 1).long().to(pts_2d.device)
#                 obj_dep_pts = obj_rect_pts[:, 2].contiguous()
#                 inds = us + vs * img_width
#                 inds = inds.contiguous()
#                 inds[inds >= img_height * img_width] = torch.randint(inds.min(), img_height * img_width - 1, (inds[inds >= img_height * img_width]).shape).to(inds.device).int()
#                 indCount = (torch.bincount(inds) > 1).int()
#                 ratio = (torch.bincount(inds) > 2).sum() / (torch.bincount(inds) > 0).sum()
#                 if ratio < 0.3:
#                     continue
#                 ''' 补0
#                 cv2.imwrite("/home/xubb/1.jpg", torch.cat(((torch.bincount(inds) > 2).int(),torch.zeros(img_height * img_width -1 -inds.max()).cuda().int()),dim=0).view(img_height, img_width).cpu().numpy()*255)
#                 '''
#                 pos_mask_left = torch.cat((indCount, torch.zeros(img_height * img_width - 1 - inds.max()).to(inds.device).int()), dim=0)
#                 cv2.imwrite("/Users/jacksonxu/Downloads/1.jpg",pos_mask_left.view(img_height, img_width).cpu().numpy() * 255)
#                 pos_mask_left = pos_mask_left > 1
#                 assert pos_mask_left.sum() > 0
#                 pos_inds_left = torch.masked_select(img_flatten_ind, pos_mask_left)

#                 depth_map = torch.zeros((img_height, img_width)).cuda()
#                 img_template = np.zeros((img_height, img_width, 3), dtype=np.uint8)
#                 for ind, el in enumerate(pos_inds_left):
#                     depth = torch.masked_select(obj_dep_pts, (inds == el)).min()
#                     depth_map[el/img_width, el%img_width] = depth
#         return points.numpy()


    