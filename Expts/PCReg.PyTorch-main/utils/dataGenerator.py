import os
import numpy
from  plot_utils import *
from pointcloud_utils import *
from file_utils import *
from process import *
from torch.utils.data import Dataset, DataLoader

class ModelNet40(Dataset):
    def __init__(self, path):
        self.path=path
        self.label,self.txt_path=self.randomFile()

    def randomDir(self,txtname="modelnet40_shape_names.txt"):
        txtpath=os.path.join(self.path,txtname)
        labels_dir=[]
        with open(txtpath,'r') as f:
            for line in f:
                labels_dir.append(line.strip())
        idx_tmp=numpy.random.uniform(0,len(labels_dir))
        label=labels_dir[int(idx_tmp)]
        # print(labels_dir)
        return label

    def randomFile(self):
        label=self.randomDir()
        filepath=os.path.join(self.path,label)
        txt_files = os.listdir(filepath)
        file_idx_tmp=np.random.randint(0,len(txt_files))
        return label,os.path.join(filepath,txt_files[file_idx_tmp])

    def randomItemGenerator(self):
        points = readTXT2Numpy(self.txt_path, diminator=',')  # (N, 6)
        points = points[:, :3]
        # visualize_pc(points,window_name=self.label)
        randomCropTwice = PointcloudRandomCropTwice()
        pts1,pts2,idx1,idx2 = randomCropTwice(points)

        # print(np.shape(pts1),'\t',np.shape(pts2),'\t',np.shape(idx1),'\t',np.shape(idx2))

        # visualize_pc(pts1,window_name=self.label)
        # visualize_pc(pts2,window_name=self.label)

        rotate_trans=PointcloudRotate()
        pts1_trans,trans_metrix=rotate_trans(torch.from_numpy(pts1))

        parral_trans=PointcloudTranslate()
        pts1_trans,trans_vector=parral_trans(pts1_trans)
        # visualize_pc(pts1_trans,window_name=self.label)

        # add Gauss noise
        pts1_trans= jitter_point_cloud(pts1_trans)
        pts2= jitter_point_cloud(pts2)
        # gauss_noise=np.random.normal(0,0.45,(np.shape(pts1_trans)))
        # pts1_trans+=gauss_noise

        return pts1_trans, pts2, idx1, idx2, trans_metrix, trans_vector

    def getDatas(self,count):
        data_sign=0
        while data_sign<count:
            self.label,self.txt_path=self.randomFile()
            pts1_trans, pts2, index1, index2, trans_metrix, trans_vector = self.randomItemGenerator()
            
            txt_file_pts1='/d2/code/datas/datas_gsNoise_txt/pts1_txt/pts1_' + str(data_sign)+'_'+self.label+'.txt'
            txt_file_pts2='/d2/code/datas/datas_gsNoise_txt/pts2_txt/pts2_' + str(data_sign)+'_'+self.label+'.txt'
            txt_index_pts1='/d2/code/datas/datas_gsNoise_txt/index1_txt/idx1_' + str(data_sign)+'_'+self.label+'.txt'
            txt_index_pts2='/d2/code/datas/datas_gsNoise_txt/index2_txt/idx2_' + str(data_sign)+'_'+self.label+'.txt'
            txt_transFile='/d2/code/datas/datas_gsNoise_txt/transform_txt/'+ 'matrix_'+str(data_sign)+'_'+self.label +'.txt'
            
            pkl_file_pts1='/d2/code/datas/datas_gsNoise/pts1/pts1_' + str(data_sign)+'_'+self.label+'.pkl'
            pkl_file_pts2='/d2/code/datas/datas_gsNoise/pts2/pts2_' + str(data_sign)+'_'+self.label+'.pkl'
            pkl_index_pts1='/d2/code/datas/datas_gsNoise/index1/idx1_' + str(data_sign)+'_'+self.label+'.pkl'
            pkl_index_pts2='/d2/code/datas/datas_gsNoise/index2/idx2_' + str(data_sign)+'_'+self.label+'.pkl'
            pkl_transFile='/d2/code/datas/datas_gsNoise/transform/'+ 'matrix_'+str(data_sign)+'_'+self.label +'.pkl'
        
            # if(np.shape(pts1_trans)[0] + np.shape(pts2)[0] <= 13000):
            print('data_sign=',data_sign,'\t','label=',self.label)
            data_sign+=1
            
            save_pickle2(pkl_file_pts1,pts1_trans)
            save_pickle2(pkl_file_pts2,pts2)
            save_pickle2(pkl_index_pts1,index1)
            save_pickle2(pkl_index_pts2,index2)
            save_pickle2(pkl_transFile,{'matrix':trans_metrix, 'vector':trans_vector})

            np.savetxt(txt_file_pts1, pts1_trans, delimiter=',',fmt='%7f')
            np.savetxt(txt_file_pts2, pts2, delimiter=',',fmt='%7f')
            np.savetxt(txt_index_pts1, index1,fmt='%5d',delimiter=',')
            np.savetxt(txt_index_pts2, index2,fmt='%5d',delimiter=',')
            np.savetxt(txt_transFile,np.vstack((trans_metrix,trans_vector)),delimiter=',')
        
        
path="/d2/code/datas/modelnet40_normal_resampled"
md40=ModelNet40(path)

# pts1,pts2,idx1,idx2,transmtx,transvec=md40.randomDataGenerator()
# # md40.randomDataGenerator()
# visualize_pc(pts1,window_name=md40.label)
# visualize_pc(pts2,window_name=md40.label)
# print(transmtx,'\n',transvec)

md40.getDatas(count=20000)


