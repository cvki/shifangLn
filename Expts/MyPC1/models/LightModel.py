
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datas.dataset import MyModelnet40
from utils.utils import compute_loss,compute_metrics
from models.Benchmark import IterativeBenchmark
from loss import EMDLosspy


class PCRegLit(pl.LightningModule):
    def __init__(self, argms):
        super().__init__()
        self.argms=argms
        self.model=IterativeBenchmark(in_dim=3, niters=argms.niters, gn=argms.gn)   # gn id False default

    def forward(self, x, y):
        batch_R_res, batch_t_res, transformed_xs=self.model(x,y)
        return batch_R_res,batch_t_res,transformed_xs

    def training_step(self, batch, batch_idx):
        loss_fn = EMDLosspy()
        pts1, pts2, R0, t0 = batch
        R_res, t_res, trans_pts = self.forward(pts1.permute(0, 2, 1).contiguous(),pts2.permute(0, 2, 1).contiguous())
        loss = compute_loss(pts2, trans_pts, loss_fn)
        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropimetrics=compute_metrics(R_res, t_res, R0, t0)
        # print('train-loss: ',loss.item())
        # print('cur_r_mse:{}, cur_r_mae:{}, cur_t_mse:{}, cur_t_mae:{}, cur_r_isotropic:{},cur_t_isotropimetrics:{}'.format(
        #         cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic,cur_t_isotropimetrics))
        return {
            'loss': loss,
            'metrics': [cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropimetrics]
        }
        return loss

    # def training_step_end(self, step_output):
    #     return step_output

    def training_epoch_end(self, outputs):
        sum_loss=0.
        sum_metrics=0.
        for item in outputs:
            print(item)


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

    def train_dataloader(self):
        return DataLoader(MyModelnet40(root=self.argms.root, mode='train', count_pcs=6000,num_pts=4096),batch_size=self.argms.batch_size)

    def val_dataloader(self):
        return DataLoader(MyModelnet40(root=self.argms.root, mode='valid', count_pcs=600,num_pts=4096),batch_size=self.argms.batch_size/2)

    def test_dataloader(self):
        return DataLoader(MyModelnet40(root=self.argms.root, mode='test'),batch_size=self.argms.batch_size/2)





if __name__=='__main__':

    # argms = config_params()
    # print(argms)

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    print(torch.cuda.current_device())

    # setup_seed(argms.seed)
    # if not os.path.exists(argms.saved_path):
    #     os.makedirs(argms.saved_path)
    # summary_path = os.path.join(argms.saved_path, 'summary')
    # if not os.path.exists(summary_path):
    #     os.makedirs(summary_path)
    # checkpoints_path = os.path.join(argms.saved_path, 'checkpoints')
    # if not os.path.exists(checkpoints_path):
    #     os.makedirs(checkpoints_path)


    # model=PCRegLit(argms)
    # trainer = pl.Trainer(default_root_dir='/d2/code/gitRes/Expts/PCReg.PyTorch-main/work_dirs_lighting/models/checkpoints',
    #                      enable_progress_bar=True,max_epochs=argms.epoches,min_epochs=30,accelerator='gpu')
    # trainer.fit(model=model,train_dataloaders=model.train_dataloader(),val_dataloaders=model.val_dataloader())


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


