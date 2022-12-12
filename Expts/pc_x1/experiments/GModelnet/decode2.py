import torch.nn.functional

from geotransformer.modules.kpconv import UnaryBlock, LastUnaryBlock
from torch import nn as nn
from torch.nn import ConvTranspose1d

class decoder2(nn.Module):
    def __init__(self,cfg):
        super(decoder2, self).__init__()
        self.dim_in=cfg.backbone.init_dim
        self.norm_group=cfg.backbone.group_norm
        self.dim_out=cfg.backbone.output_dim
        self.dec1 = UnaryBlock(self.dim_in * 4, self.dim_in * 8, self.norm_group)
        self.dec2 = UnaryBlock(self.dim_in * 8, self.dim_in * 16, self.norm_group)
        self.dec3 = LastUnaryBlock(self.dim_in * 16, self.dim_out * 32)
        self.transConv=ConvTranspose1d((self.dim_in * 4, self.dim_in * 8, self.norm_group))
        # self.subsampling_mlp = nn.Linear(self.dim_in * 16, self.dim_in * 8, bias=True)

    # def forward(self,feat_list):
    def forward(self,feats):

        # feat_res=[]
        #
        # # add bottle and ist decoder
        # # feats_de1 = feat_list[-1]
        # feats_de1 = feats
        # feats_de1 = self.dec1(feats_de1)
        # # latent_s4 = self.max_pooling(latent_s4_tmp)
        # # feat_res.append(feats_de1+feat_list[-2])
        # feat_res.append(feats_de1)
        #
        # # feats_de2 = feat_list[-2]
        # feats_de2 = self.dec2(feats_de1)
        # # feat_res.append(feats_de2+feat_list[-3])
        # feat_res.append(feats_de2)
        #
        # # feats_de3 = feat_list[-3]
        # feats_de3 = self.dec3(feats_de2)
        # # feat_res.append(feats_de3+feat_list[-4])
        # feat_res.append(feats_de3)
        #
        # feats_de3.
        #
        # return feat_res
        pass


class decoder_up(nn.Module):
    def __init__(self,cfg):
        super(decoder_up, self).__init__()
        self.convTrans4 = ConvTranspose1d(cfg.backbone.init_dim*4,cfg.backbone.init_dim*8,kernel_size=3)
        self.convTrans3 = ConvTranspose1d(cfg.backbone.init_dim*8,cfg.backbone.init_dim*16,kernel_size=3)
        self.convTrans2 = ConvTranspose1d(cfg.backbone.init_dim*16,cfg.backbone.init_dim*32,kernel_size=3)
        self.convTrans1 = ConvTranspose1d(cfg.backbone.init_dim*32,cfg.backbone.init_dim*64,kernel_size=3)
        self.mlp=nn.Linear(cfg.backbone.init_dim*4+8,cfg.backbone.init_dim*2)

    def forward(self,feats):
        feats=self.convTrans4(feats)
        feats=self.convTrans3(feats)
        feats=self.convTrans2(feats)
        feats=self.convTrans1(feats)
        feats=self.mlp(feats)
        return torch.nn.functional.leaky_relu(feats)







