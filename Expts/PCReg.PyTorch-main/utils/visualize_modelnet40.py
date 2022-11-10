import os;os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from file_utils import *
from utils.pointcloud_utils import *
from plot_utils import *
import numpy as np

all_npzs = make_dataset('/Users/jacksonxu/Downloads/modelnet40_test_pkl_k30_l2', suffix='.npz')
classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
for npz_file in all_npzs:
    info = np.load(npz_file)
    points, point_scores, label = info['points'], info['point_scores'], info['label'] # point_scores \in (0,1)
    colors = point_scores[:, np.newaxis].repeat(3, axis=-1) * np.array([[1.0,0.0,0.0]])
    colors = (colors*255).astype(np.uint8) # red is important
    visualize_pc(points, colors, window_name=classes[int(label)])