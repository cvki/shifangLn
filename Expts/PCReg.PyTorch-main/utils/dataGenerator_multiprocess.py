import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import numpy
from utils.plot_utils import *
from utils.pointcloud_utils import *
from utils.file_utils import *
from utils.process import *
from torch.utils.data import Dataset, DataLoader

# label,txt_path=randomFile()

def randomDir(txtname="modelnet40_shape_names.txt"):
    txtpath=os.path.join(path,txtname)
    labels_dir=[]
    with open(txtpath,'r') as f:
        for line in f:
            labels_dir.append(line.strip())
    idx_tmp=numpy.random.uniform(0,len(labels_dir))
    label=labels_dir[int(idx_tmp)]
    # print(labels_dir)
    return label

def randomFile():
    label=randomDir()
    filepath=os.path.join(path,label)
    txt_files = os.listdir(filepath)
    file_idx_tmp=np.random.randint(0,len(txt_files))
    return label,os.path.join(filepath,txt_files[file_idx_tmp])

def randomItemGenerator(txt_path):
    points = readTXT2Numpy(txt_path, diminator=',')  # (N, 6)
    points = points[:, :3]
    randomCropTwice = PointcloudRandomCropTwice()
    pts1,pts2,idx1,idx2 = randomCropTwice(points)
    if pts1 is None:
        return None, None, None, None, None, None

    # print(np.shape(pts1),'\t',np.shape(pts2),'\t',np.shape(idx1),'\t',np.shape(idx2))

    # visualize_pc(pts1,window_name=self.label)
    # visualize_pc(pts2,window_name=self.label)

    rotate_trans=PointcloudRotate()
    pts1_trans,trans_metrix=rotate_trans(torch.from_numpy(pts1))
    quat_vec=mat2quat(trans_metrix)

    parral_trans=PointcloudTranslate()
    pts1_trans,trans_vector=parral_trans(pts1_trans)
    # visualize_pc(pts1_trans,window_name=self.label)

    # add Gauss noise
    pts1_trans= jitter_point_cloud(pts1_trans)
    pts2= jitter_point_cloud(pts2)
    # gauss_noise=np.random.normal(0,0.45,(np.shape(pts1_trans)))
    # pts1_trans+=gauss_noise
    quat_vec=np.append(quat_vec,trans_vector).reshape(1,7)

    return pts1_trans, pts2, idx1, idx2, trans_metrix, trans_vector, quat_vec


def process(data_sign):
    np.random.seed(data_sign)   # seed
    label,txt_path=randomFile()
    # pts1_trans, pts2, index1, index2, trans_metrix, trans_vector, quat = randomItemGenerator(txt_path)
    try:
        pts1_trans, pts2, index1, index2, trans_metrix, trans_vector, quat = randomItemGenerator(txt_path)
        # pts1_trans, pts2, index1, index2, trans_metrix, trans_vector, quat = randomItemGenerator('/d2/code/datas/modelnet40_normal_resampled/radio/radio_0057.txt')
    except:
        # print(txt_path)
        pts1_trans = None
    if pts1_trans is None:
        return
    
    pkl_file_pts=pkl_path+ 'pts_'+str(data_sign)+'.pkl'
    # txt_file_pts1='/d2/code/datas/datas_gsNoise_txt/pts1_txt/pts1_' + str(data_sign)+'_'+label+'.txt'
    # txt_file_pts2='/d2/code/datas/datas_gsNoise_txt/pts2_txt/pts2_' + str(data_sign)+'_'+label+'.txt'
    # txt_index_pts1='/d2/code/datas/datas_gsNoise_txt/index1_txt/idx1_' + str(data_sign)+'_'+label+'.txt'
    # txt_index_pts2='/d2/code/datas/datas_gsNoise_txt/index2_txt/idx2_' + str(data_sign)+'_'+label+'.txt'
    # txt_transFile='/d2/code/datas/datas_gsNoise_txt/transform_txt/'+ 'matrix_'+str(data_sign)+'_'+label +'.txt'
    
    # pkl_file_pts1='/d2/code/datas/datas_gsNoise/pts1/pts1_' + str(data_sign)+'_'+label+'.pkl'
    # pkl_file_pts2='/d2/code/datas/datas_gsNoise/pts2/pts2_' + str(data_sign)+'_'+label+'.pkl'
    # pkl_index_pts1='/d2/code/datas/datas_gsNoise/index1/idx1_' + str(data_sign)+'_'+label+'.pkl'
    # pkl_index_pts2='/d2/code/datas/datas_gsNoise/index2/idx2_' + str(data_sign)+'_'+label+'.pkl'
    # pkl_transFile='/d2/code/datas/datas_gsNoise/transform/'+ 'matrix_'+str(data_sign)+'_'+label +'.pkl'

    # if(np.shape(pts1_trans)[0] + np.shape(pts2)[0] <= 13000):
    # print('data_sign=',data_sign,'\t','label=',label)
    # data_sign+=1
    
   
    # save_pickle2(pkl_file_pts2,pts2)
    # save_pickle2(pkl_index_pts1,index1)
    # save_pickle2(pkl_index_pts2,index2)
    # save_pickle2(pkl_transFile,{'matrix':trans_metrix, 'vector':trans_vector})

    # np.savetxt(txt_file_pts1, pts1_trans, delimiter=',',fmt='%7f')
    # np.savetxt(txt_file_pts2, pts2, delimiter=',',fmt='%7f')
    # np.savetxt(txt_index_pts1, index1,fmt='%5d',delimiter=',')
    # np.savetxt(txt_index_pts2, index2,fmt='%5d',delimiter=',')
    # np.savetxt(txt_transFile,np.vstack((trans_metrix,trans_vector)),delimiter=',')
    
    save_pickle2(pkl_file_pts, 
            {'pts1': pts1_trans, 'pts2': pts2, 'idx1': index1, 'idx2': index2, 'trans_mat': trans_metrix, 'trans_vec': trans_vector, 
             'quat_vec': quat,'label':label})




if __name__=='__main__':
    pkl_path='/d2/code/datas/MyModelNet40/valid/' 
    path="/d2/code/datas/modelnet40_normal_resampled"

    # txt_root = '/d2/code/datas/datas_gsNoise_txt/'
    # remove_and_mkdir(txt_root)
    # pkl_root = '/d2/code/datas/datas_gsNoise/'
    # pkl_root = '/d2/code/datas/datas_gsNoise/'

    # mkdir_if_no(os.path.join(txt_root, 'pts1_txt'))
    # mkdir_if_no(os.path.join(txt_root, 'pts2_txt'))
    # mkdir_if_no(os.path.join(txt_root, 'index1_txt'))
    # mkdir_if_no(os.path.join(txt_root, 'index2_txt'))
    # mkdir_if_no(os.path.join(txt_root, 'transform_txt'))

    # process(0)

    # remove_and_mkdir(pkl_path)

    # run_imap_multiprocessing(process, range(30000), 16)  # train---30000
    # run_imap_multiprocessing(process, range(6000), 8)  # valid---6000
    # run_imap_multiprocessing(process, range(6000), 8)  # test---6000

