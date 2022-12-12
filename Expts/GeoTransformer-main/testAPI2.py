# import torch
#
# a1=torch.randint(0,10,(3,5))
# print(a1)
# val,id=a1.topk(k=2,dim=1,largest=False)
# print(val,'\n',id)

'''关于该模型的一些问题（或特点）：
    1. correspondence groundtruth 的创建过程（作为邻域的一种方法：superpoints思想(node,points)，node-node和node-points强关联，patch(node,points)匹配方法）
    2. transformer（self和cross --attention）在点云内部和点云之间的使用，结合(gcn)'Predator'。点云无序性特点和transformer的天然结合。
    3. embedding：从distance和angles角度，背后的数学原因
    4. superpoints-matching中，Gaussian correlation matrix相对于Li-distance的区别
    5.
    6.其他：inlines(overlaps)的角度<correspondences>，还有非基于correspondence的方式
'''