import math
import torch
import numpy as np


def isPrime(num: int):
    border = int(math.sqrt(float(num)))
    for i in range(2, border):
        if num % i == 0:
            return False
    return True


def calNextPrime(num: int):
    while not isPrime(num):
        num += 1
    return num


class Model_Args(object):
    def __init__(
        self,
        n_extra_layers,
        d_model,
        dropout,
        share_dim,
        patience,
        batch_size,
        num_workers,
        learning_rate,
        train_epochs
    ):
        self.e_layers = n_extra_layers
        self.d_model = d_model
        self.dropout = dropout
        self.d_share = share_dim
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs

    def update_size(self, width, depth):
        self.bucket_dim = width
        self.hash_num = depth

    def update_path(self, ckpt_path):
        self.checkpoints = ckpt_path

    def update_interval(self, interval):
        self.interval = interval

    def update_gpu(self, use_gpu, use_multi_gpu=False, gpu_id=0):
        self.use_multi_gpu = use_multi_gpu
        self.use_gpu = use_gpu
        self.gpu = gpu_id

    def select_ablation(self, ablation_type):
        self.ablation = ablation_type


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta * self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pt')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {100: args.learning_rate * 0.5 ** 1, 150: args.learning_rate * 0.5 ** 2}
    elif args.lradj=='type2':
        lr_adjust = {50: args.learning_rate * 0.5 ** 1, 100: args.learning_rate * 0.5 ** 2,
                     150: args.learning_rate * 0.5 ** 3, 200: args.learning_rate * 0.5 ** 4,
                     250: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))