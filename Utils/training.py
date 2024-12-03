import os
import time
import torch
import warnings
import numpy as np

from torch import nn
from torch import optim
from torch.nn import DataParallel
from load_data import sketchDataset
from torch.utils.data import DataLoader
from Utils.common import Exp_Basic, EarlyStopping
from net_params import inverseNet, inverseNet_ablation

warnings.filterwarnings('ignore')


class learningSolver(Exp_Basic):
    def __init__(self, args, out_dim):
        self.out_dim = out_dim
        super(learningSolver, self).__init__(args)
    
    def _build_model(self):
        if self.args.ablation != 1:
            model = inverseNet(
                self.args.e_layers,
                self.args.bucket_dim,
                self.args.hash_num,
                self.args.d_model,
                self.out_dim,
                dropout=self.args.dropout,
                share_dim=self.args.d_share
            ).float()
        else:
            model = inverseNet_ablation(
                self.args.e_layers,
                self.args.bucket_dim,
                self.args.hash_num,
                self.args.d_model,
                self.out_dim,
                dropout=self.args.dropout,
                share_dim=self.args.d_share
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, data, A=None):
        args = self.args

        if A is not None:
            self.A = torch.from_numpy(A).float().to(self.device)
        
        if data.shape[0] == 1:
            drop_last = False
        else:
            drop_last = True
        
        data_set = sketchDataset(data)
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_loader

    def _select_optimizer(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def intance_norm(self, inp):
        scale = torch.min(torch.max(inp, dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]
        return inp / scale, scale

    def transform(self, inp, ind, add_total=100):
        inp = torch.clamp_min(inp, 1.)
        len = inp.shape[-1]
        origin = inp[:, ind]
        zeros = torch.zeros_like(inp)
        add_indices = np.random.choice(
            np.ones(len), (int)(len * 0.05), replace=False
        )
        origin_prob = origin / torch.sum(origin, dim=-1, keepdim=True)
        zeros[:, ind] = origin_prob * add_total
        zeros[:, add_indices] = zeros[:, add_indices] + 1
        return inp + zeros

    def train(self, sketchShots, phiMatrix, index, weight=0.1):
        index = torch.from_numpy(np.array(index)).long().to(self.device)
        train_loader = self._get_data(sketchShots, phiMatrix)

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.01)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_x, scale = self.intance_norm(batch_x)

                batch_x = batch_x.reshape(batch_x.shape[0], -1)
                rec_y = self.model(batch_x)

                gen_x = (self.A @ rec_y.transpose(0, 1)).transpose(0, 1)
                loss_x = criterion(gen_x, batch_x)
                
                y = rec_y * scale.squeeze(1)
                y = self.transform(y, index, self.args.interval).detach()
                gen_x_new = (self.A @ y.transpose(0, 1)).transpose(0, 1)
                batch_new, scale = self.intance_norm(gen_x_new.reshape(gen_x_new.shape[0], self.args.hash_num, -1))
                batch_new = batch_new.reshape(batch_new.shape[0], -1)
                gen_y = self.model(batch_new)
                loss_y = criterion(gen_y, y / scale.squeeze(1))

                loss_reg = weight * torch.mean(rec_y)

                a = 0 if self.args.ablation == 2 else 1
                b = 0 if self.args.ablation == 3 else 1

                loss = loss_x + a * loss_y + b * loss_reg
                train_loss.append(loss.item())

                if self.args.ablation == 0 and i % 100 == 0 and i != 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}, loss_key: {3:.4f}, loss_val: {4:.4f}"
                          .format(i + 1, epoch + 1, loss_reg.item(), loss_x.item(), loss_y.item()))
                    speed = (time.time()-time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pt'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pt')


    def test(self, sketchShots, save_pred=False):
        test_loader = self._get_data(sketchShots)
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for i, batch_x in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                scale = torch.min(torch.max(batch_x, dim=-1)[0], dim=-1)[0]
                batch_x /= scale

                batch_x = batch_x.reshape(batch_x.shape[0], -1)
                rec_y = self.model(batch_x) * scale
                    
        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            np.save(folder_path + 'pred.npy', preds)

        return rec_y.detach().cpu().numpy()
