from itertools import count
from torch.utils.data import Dataset
import os
import pickle
import torch
from utils import load_pickle

# data_path='/d2/code/datas/vs_modelnet40/' 

class MyModelnet40(Dataset):
    def __init__(self,dir,mode,count_pcs=None,num_pts=None) -> None:
        self.path=os.listdir(os.path.join(dir,mode))
        self.filenames=os.listdir(self.path)
        self.count_pcs=count_pcs
        self.num_pts=num_pts
    def __getitem__(self, index):
        file=os.path.join(self.path,self.filenames[index])
        datas=load_pickle(file)
        pts1,pts2,R,t,qutanr=datas['pts1'],datas['pts2'],datas['trans_mat'],datas['trans_vec'],datas['quat_vec']
        if not self.num_pts is None:     # when there're too many datas in a pickle: sampling
            pts1=pts1[torch.randint(len(self.filenames),self.num_pts)] 
            pts2=pts2[torch.randint(len(self.filenames),self.num_pts)] 
        return pts1,pts2,R,t
    
    def __len__(self):      # when there're too many pickle files in a dir: sampling. Bug:only use the top count_pcs files
        return len(self.filenames) if self.count_pcs is None or self.count_pcs>len(self.filenames) else count
        
        

        
        