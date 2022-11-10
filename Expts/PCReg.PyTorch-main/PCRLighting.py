import numpy as np
import pytorch_lightning as pl
import torch
import os
from torch.utils.data import DataLoader
from data.ModelNet40 import MyModelNet40
from modelnet40_train import config_params,compute_loss,setup_seed
from models.benchmark import IterativeBenchmark
from loss import EMDLosspy
from metrics.helper import compute_metrics
import open3d as o3d
from utils.format import npy2pcd
from utils.file_utils import load_pickle



# def show_checkpoints_pcd(arg_test,x, y):
#     model_checkpoint = PCRegLit(arg_test)
#     state_dict = torch.load('/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints/lightning_logs/version_23/checkpoints/epoch=59-step=45000.ckpt')['state_dict']
#     model_checkpoint.load_state_dict(state_dict)
#
#     # model_checkpoint = PCRegLit(arg_test).load_from_checkpoint(
#     #     checkpoint_path='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints/lightning_logs/version_23/checkpoints/epoch=59-step=45000.ckpt',
#     #     # hparams_file='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints/lightning_logs/version_23/hparams.yaml'
#     #     # map_location=lambda storage, loc: storage
#     # )
#
#     with torch.no_grad():
#         R, t, trans_pcds = model_checkpoint(x, y)
#         for trans_pcd in trans_pcds:
#             x = torch.squeeze(x).cpu().numpy()
#             y = torch.squeeze(y).cpu().numpy()
#             trans_pcd = torch.squeeze(trans_pcd[-1]).cpu().numpy()
#             pcd1 = npy2pcd(x, 0)
#             pcd2 = npy2pcd(y, 1)
#             pcd3 = npy2pcd(trans_pcd, 2)
#             o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
#
# def show_test():
#     test_path='/d2/code/datas/MyModelNet40/test'
#     files=os.listdir(test_path)
#     for f1 in files:
#         file=load_pickle(os.path.join(test_path,f1))
#         # pts1,pts2=datas['pts2'],datas['pts1']
#         x,y=file['pts2'][:4096,:],file['pts1'][:4096,:]
#         show_checkpoints_pcd(torch.from_numpy(x),torch.from_numpy(y))



class PCRegLit(pl.LightningModule):
    def __init__(self, argms):
        super().__init__()
        # self.root_path=argms.root
        # self.epoches=argms.epoches
        # self.niters=argms.niters
        # self.lr=argms.lr
        # self.saved_path=argms.saved_path
        # self.saved_frequency=argms.saved_frequency
        self.argms=argms
        self.model=IterativeBenchmark(in_dim=3, niters=argms.niters, gn=argms.gn)

    def forward(self, x, y):
        batch_R_res, batch_t_res, transformed_xs=self.model(x,y)
        return batch_R_res,batch_t_res,transformed_xs

    def training_step(self, batch, batch_idx):
        loss_fn = EMDLosspy()
        # pts1, pts2, R0, t0 = self.train_dataloader()
        pts1, pts2, R0, t0 = batch
        R_res, t_res, trans_pts = self.forward(pts1.permute(0, 2, 1).contiguous(),pts2.permute(0, 2, 1).contiguous())
        loss = compute_loss(pts2, trans_pts, loss_fn)
        # cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropimetrics=compute_metrics(R_res, t_res, R0, t0)
        # print('train-loss: ',loss.item())
        # print('cur_r_mse:{}, cur_r_mae:{}, cur_t_mse:{}, cur_t_mae:{}, cur_r_isotropic:{},cur_t_isotropimetrics:{}'.format(
        #         cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic,cur_t_isotropimetrics))
        # return {
        #     'loss': loss,
        #     'metrics': [cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropimetrics]
        # }
        return loss

    def training_epoch_end(self,losses):
        # self.log('train-loss',torch.mean(torch.Tensor(losses)))
        # self.log('train-loss', 0.2)
        # print('losses',type(losses))
        # self.log('test-loss',0.5)
        sum_loss = 0.
        for loss in losses:
            sum_loss += loss['loss']
            # pass
        print('train-loss: ', sum_loss / len(losses))
        # print('-----------------------train',losses)


    def validation_step(self, batch, batch_idx):
        loss_fn = EMDLosspy()
        pts1, pts2, R0, t0 =batch
        R_res, t_res, trans_pts = self.forward(pts1.permute(0, 2, 1).contiguous(), pts2.permute(0, 2, 1).contiguous())
        loss = compute_loss(pts2, trans_pts, loss_fn)
        return loss
        # cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropimetrics = compute_metrics(R_res, t_res, R0, t0)
        # result=[
        #     {'loss': loss},
        #     {'criterion': {'cur_r_mse': cur_r_mse,
        #                     'cur_r_mae': cur_r_mae,
        #                     'cur_t_mse': cur_t_mse,
        #                     'cur_t_mae': cur_t_mae,
        #                     'cur_r_isotropic': cur_r_isotropic,
        #                     'cur_t_isotropimetrics': cur_t_isotropimetrics
        #                    }
        #      }
        # ]

        # result={
        #     'loss': loss,
        #     # 'cur_r_mse': torch.mean(cur_r_mse).item(),
        #     # 'cur_r_mae': torch.mean(cur_r_mae).item(),
        #     # 'cur_t_mse': torch.mean(cur_t_mse).item(),
        #     # 'cur_t_mae': torch.mean(cur_t_mae).item(),
        #     # 'cur_r_isotropic': torch.mean(cur_r_isotropic).item(),
        #     # 'cur_t_isotropimetrics': torch.mean(cur_t_isotropimetrics).item()
        # }

        # self.log('test-loss', loss.item())
        # self.log('cur_r_mse', cur_r_mse)
        # self.log('cur_r_mae', cur_r_mae)
        # self.log('cur_t_mse', cur_t_mse)
        # self.log('cur_t_mae', cur_t_mae)
        # self.log('cur_r_isotropic', cur_r_isotropic)
        # self.log('cur_t_isotropimetrics',cur_t_isotropimetrics)
        # self.log_dict(result ,prog_bar=True)
        # return result

    # def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
    # def validation_epoch_end(self) -> None:

    def validation_epoch_end(self,losses):
        # self.log('val-loss',torch.mean(torch.Tensor(losses)))
        # print('losses',type(losses))
        # self.log('test-loss',0.5)
        sum_loss=0.
        for loss in losses:
            sum_loss+=loss
        print('val-loss: ',sum_loss/len(losses))
        # print('-----------------------val',losses)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.argms.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.argms.milestones, gamma=self.argms.gamma,
                                                         last_epoch=-1)
        return ([optimizer],[scheduler])
        # return optimizer, scheduler   # ERROR

    ####################
    # DATA RELATED HOOKS
    ####################

    # def prepare_data(self):
    #     # download or datagenerator
    #     train_set = MyModelNet40(root=self.argms.root, train_mode='train', normal=self.argms.normal, mode=self.argms.mode, size=60)
    #     val_set = MyModelNet40(root=self.argms.root, train_mode='valid', normal=self.argms.normal, mode=self.argms.mode, size=6)
    #     test_set = MyModelNet40(root=self.argms.root, train_mode='test', normal=self.argms.normal, mode=self.argms.mode)
    #     # print('train_set shape:{},valid_set shape:{},test_set shape:{}'.format(np.shape(train_set),np.shape(val_set),np.shape(test_set)))
    #     return [train_set, val_set, test_set]

    # def setup(self, stage=None):
    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "train" or stage is None:
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
    #
    #     # Assign test dataset for use in dataloader(s)
    #     elif stage == "valid":
    #         self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    #     elif stage=="test":
    #         pass

    def train_dataloader(self):
        return DataLoader(MyModelNet40(root=self.argms.root, train_mode='train', normal=self.argms.normal, mode=self.argms.mode, size=6000),batch_size=8)

    def val_dataloader(self):
        return DataLoader(MyModelNet40(root=self.argms.root, train_mode='valid', normal=self.argms.normal, mode=self.argms.mode, size=600),batch_size=4)

    def test_dataloader(self):
        return DataLoader(MyModelNet40(root=self.argms.root, train_mode='test', normal=self.argms.normal, mode=self.argms.mode),batch_size=1)





if __name__=='__main__':

    argms = config_params()
    print(argms)

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    print(torch.cuda.current_device())

    setup_seed(argms.seed)
    if not os.path.exists(argms.saved_path):
        os.makedirs(argms.saved_path)
    summary_path = os.path.join(argms.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(argms.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)


    model=PCRegLit(argms)
    trainer = pl.Trainer(default_root_dir='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints',
                         enable_progress_bar=True,max_epochs=argms.epoches,min_epochs=30,accelerator='gpu')
    trainer.fit(model=model,train_dataloaders=model.train_dataloader(),val_dataloaders=model.val_dataloader())


    # def show_checkpoints_pcd(x, y):
    #     model_checkpoint=PCRegLit.load_from_checkpoint('/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints/lightning_logs/version_23/checkpoints',
    #                                   hparams_file='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints/lightning_logs/version_23/hparams.yaml')
    #     R,t,trans_pcds=model_checkpoint(x,y)
    #     for trans_pcd in trans_pcds:
    #         from utils.format import npy2pcd
    #         import open3d as o3d
    #         x = torch.squeeze(x).cpu().numpy()
    #         y = torch.squeeze(y).cpu().numpy()
    #         trans_pcd = torch.squeeze(trans_pcd[-1]).cpu().numpy()
    #         pcd1 = npy2pcd(x, 0)
    #         pcd2 = npy2pcd(y, 1)
    #         pcd3 = npy2pcd(trans_pcd, 2)
    #         o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])


