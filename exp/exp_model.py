from data.data_loader import Dataset_ETT_hour, Dataset_ETT_min, Dataset_Custom
from exp.exp_basic import Exp_Basic
from FPPformer2.FPPformer import FPPformer
from FPPformer2.FPPformer_Cross import FPPformer_Cross

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        if self.args.Cross:
            model = FPPformer_Cross(
                self.args.input_len,
                self.args.pred_len,
                self.args.encoder_layer,
                self.args.patch_size,
                self.args.d_model,
                self.args.dropout
            ).float()
        else:
            model = FPPformer(
                self.args.input_len,
                self.args.pred_len,
                self.args.encoder_layer,
                self.args.patch_size,
                self.args.d_model,
                self.args.dropout
            ).float()
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_min,
            'ETTm2': Dataset_ETT_min,
            'weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'Traffic': Dataset_Custom,
            'Air': Dataset_Custom,
            'River': Dataset_Custom,
            'BTC': Dataset_Custom,
            'ETH': Dataset_Custom
        }
        Data = data_dict[self.args.data]

        size = [args.input_len, args.pred_len]
        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=size,
            features=args.features,
            target=args.target,
            ori_target=args.ori_target,
            EMD=args.EMD
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, pred_x, var_mask) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                loss = torch.mean((pred - true) ** 2).detach().cpu().numpy()
                total_loss.append(loss)
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        lr = self.args.learning_rate

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, pred_x, var_mask) in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                loss = torch.mean((pred - true) ** 2) + torch.mean(abs(pred - true))
                loss.backward(loss)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} Test Loss: {4:.7f}| "
                  .format(self.args.pred_len, epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, (epoch + 1), self.args)

        self.args.learning_rate = lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False, write_loss=True, save_loss=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()
        if save_loss:
            preds = []
            trues = []
            with torch.no_grad():
                for i, (batch_x, pred_x, var_mask) in enumerate(test_loader):
                    pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)

            print("inference time: {}".format(time.time() - time_now))
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)
            mae, mse = metric(preds, trues)
            print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + f'metrics.npy', np.array([mae, mse]))
            np.save(folder_path + f'pred.npy', preds)
            np.save(folder_path + f'true.npy', trues)

        else:
            mses = []
            maes = []
            with torch.no_grad():
                for i, (batch_x, pred_x, var_mask) in enumerate(test_loader):
                    pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    mae = np.mean(abs(pred - true))
                    mse = np.mean((pred - true) ** 2)
                    maes.append(mae)
                    mses.append(mse)

            print("inference time: {}".format(time.time() - time_now))
            mae = np.mean(maes)
            mse = np.mean(mses)
            print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')

        if write_loss:
            path = './result.log'
            with open(path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                f.write('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                        format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')
                f.flush()
                f.close()
        else:
            pass

        if not save_loss:
            dir_path = os.path.join(self.args.checkpoints, setting)
            check_path = dir_path + '/' + 'checkpoint.pth'
            if os.path.exists(check_path):
                os.remove(check_path)
                os.removedirs(dir_path)
        return mse, mae

    def _process_one_batch(self, batch_x, pred_x, var_mask):
        batch_x = batch_x.float().to(self.device)
        input_seq = batch_x[:, :self.args.input_len, :]
        batch_y = batch_x[:, -self.args.pred_len:, :]
        pred_x = pred_x.float().to(self.device)
        var_mask = var_mask.bool().to(self.device)
        pred_data = self.model(input_seq, pred_x, var_mask)
        return pred_data, batch_y
