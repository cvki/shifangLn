

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())

# t_tm=torch.rand(2,3)
# print(t_tm)
# print(t_tm.requires_grad)

# x_tm=torch.arange(4,dtype=torch.float32,requires_grad=True)
# print(x_tm)
# print(x_tm.requires_grad)

# x_tm=torch.arange(4,dtype=torch.uint8,requires_grad=True)   
# # ERROR, only floating points and conplex dtype require gradients
# print(x_tm)
# print(x_tm.requires_grad)



from torch.utils.data import DataLoader
import torch

from dataset import MyModelnet40

dir='/d2/code/datas/vs_modelnet40'
train_loader=DataLoader(MyModelnet40(dir=dir,mode='train',count_pcs=4096,num_pts=4096))
valid_loader=DataLoader(MyModelnet40(dir=dir,mode='valid',count_pcs=4096,num_pts=4096))
test_loader=DataLoader(MyModelnet40(dir=dir,mode='test',count_pcs=4096,num_pts=4096))

def train():
    pass
def test():
    pass

