import torch
from model import GeoTransformer
from config import make_cfg

if __name__=='__main__':
    cfg=make_cfg()
    params =torch.load(
        '/d2/code/gitRes/Expts/GeoTransformer-222/output/'  \
        'geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/iter-290000.pth.tar')
    model=GeoTransformer(cfg)
    model.load_state_dict(params['model'])
    print(model)
